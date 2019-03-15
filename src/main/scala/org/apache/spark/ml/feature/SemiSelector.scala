package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, UnresolvedAttribute}
import org.apache.spark.ml.feature.SemiSelectorModel.SemiSelectorWriter
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.storage.StorageLevel

private[feature] trait SemiSelectorParams extends Params
  with HasFeaturesCol with HasOutputCol with HasLabelCol with DefaultParamsWritable {

  final val numTopFeatures = new IntParam(this, "numTopFeatures",
    "Number of features that selector will select.",
    ParamValidators.gtEq(1))
  setDefault(numTopFeatures, 20)

  final val delta = new DoubleParam(this, "delta",
    "The threshold of neighborhood relationship.",
    ParamValidators.gt(0)
  )

  // TODO param of label missing value name
}


class SemiSelector(override val uid: String)
  extends Estimator[SemiSelectorModel] with SemiSelectorParams {

  def this() = this(Identifiable.randomUID("semiSelector"))

  def setNumTopFeatures(value: Int): this.type = set(numTopFeatures, value)

  def setDelta(value: Double): this.type = set(delta, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def fit(dataset: Dataset[_]): SemiSelectorModel = {
    transformSchema(dataset.schema)
    // if success, numAttributes well be None while attributes will contain info.
    val attributeGroup = AttributeGroup.fromStructField(dataset.schema($(featuresCol)))
    val attr = attributeGroup.attributes.get
    val numAttributes = attr.length
    val nominalIndices = attr.filter(_.isNominal).map(_.index.get).toSet
    // TODO get missing value info
    val label = Attribute.fromStructField(dataset.schema($(labelCol)))
    // transform origin data into dataset can be handle by DistributeNeighborEntropy.
    val data = dataset.select(col($(labelCol)), col($(featuresCol)))
    val rotatedData = NeighborEntropyHelper.rotateDFasRDD(data, 10)
    val bcNominalIndices = data.sparkSession.sparkContext.broadcast(nominalIndices)
    val formattedData = NeighborEntropyHelper.formatData(rotatedData, bcNominalIndices)
    formattedData.persist(StorageLevel.MEMORY_ONLY)

    val attrCol = formattedData.filter(_._1 != numAttributes)
    val classCol = formattedData.filter(_._1 == numAttributes).first

    val neighborEntropy = new DistributeNeighborEntropy(
      $(delta),
      formattedData,
      bcNominalIndices
    )
    val selected = collection.mutable.Set[Int]()
    // TODO implement a semi-supervised version
    // 这里可以创建一个Rdd过滤掉无类标数据
    val relevance = neighborEntropy.mutualInformation(classCol, attrCol).toMap
    val accumulateRedundancy = Array.ofDim[Double](numAttributes)
    var currentSelected = relevance.maxBy(_._2)
    selected += currentSelected._1
    while (selected.size < $(numTopFeatures)) {
      val candidateCol = attrCol.filter(col => !selected.contains(col._1))
      val previousCol = attrCol.filter(_._1 == currentSelected._1).first()
      val mis = neighborEntropy.mutualInformation(previousCol, candidateCol)
      mis.foreach { case (index, mi) => accumulateRedundancy(index) += mi }
      val selectedCounts = selected.size
      val measures = mis.map {
        case (index, _) =>
          val sig = relevance(index) - accumulateRedundancy(index) / selectedCounts
          (index, sig)
      }
      currentSelected = measures.maxBy(_._2)
      selected += currentSelected._1
    }
    // Broadcast should be removed after it is determined that it is no longer need(include a RDD recovery compute).
    bcNominalIndices.destroy()
    new SemiSelectorModel(uid, selected.toArray)
  }


  override def copy(extra: ParamMap): Estimator[SemiSelectorModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.checkNumericType(schema, $(labelCol))
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

object SemiSelector extends DefaultParamsReadable[SemiSelector] {
  override def load(path: String): SemiSelector = super.load(path)
}

final class SemiSelectorModel private[ml](
    override val uid: String,
    val selectedFeatures: Array[Int]) extends Model[SemiSelectorModel] with SemiSelectorParams with MLWritable {
  private val filterIndices = selectedFeatures.sorted

  override def copy(extra: ParamMap): SemiSelectorModel = {
    val copied = new SemiSelectorModel(uid, selectedFeatures)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new SemiSelectorWriter(this)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformedSchema = transformSchema(dataset.schema)
    val newField = transformedSchema.last
    val slicer = udf { vec: Vector =>
      vec match {
        case features: DenseVector => Vectors.dense(filterIndices.map(features(_)))
        case features: SparseVector => features.slice(filterIndices)
      }
    }
    dataset.withColumn($(outputCol), slicer(col($(featuresCol))), newField.metadata)
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    val newField = prepOutputField(schema)
    StructType(schema :+ newField)
  }

  def prepOutputField(schema: StructType): StructField = {
    val origAttrGroup = AttributeGroup.fromStructField(schema($(featuresCol)))
    // transformSchema method will run two times in a pipeline.
    // First, PipelineModel has a transformSchema which will invoke each stage's method.
    // Second, when perform a dataset transform of each stage, transformSchema will perform.
    val selectAttr = if (origAttrGroup.attributes.nonEmpty) {
      // For the second invoke, we need schema.
      origAttrGroup.attributes.get.zipWithIndex.filter(a => filterIndices.contains(a._2)).map(_._1)
    } else {
      // For the first invoke, we only need know the type of output not the schema.
      // And because when a transformSchema invoke by the Pipeline's, none of stage really run. There is no way to get
      // schema.
      Array.fill[Attribute](filterIndices.length)(UnresolvedAttribute)
    }
    val newAttrGroup = new AttributeGroup($(outputCol), selectAttr)
    newAttrGroup.toStructField
  }
}

object SemiSelectorModel extends MLReadable[SemiSelectorModel] {

  private[SemiSelectorModel] class SemiSelectorWriter(instance: SemiSelectorModel) extends MLWriter {

    private case class Data(selectedFeatures: Seq[Int])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.selectedFeatures.toSeq)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class SemiSelectorReader extends MLReader[SemiSelectorModel] {
    private val className = classOf[SemiSelectorModel].getName

    override def load(path: String): SemiSelectorModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath).select("selectedFeatures").head
      val selectedFeatures = data.getAs[Seq[Int]](0).toArray
      val model = new SemiSelectorModel(metadata.uid, selectedFeatures)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  override def read: MLReader[SemiSelectorModel] = new SemiSelectorReader

  override def load(path: String): SemiSelectorModel = super.load(path)
}