package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.feature.NmiSelectorModel.NmiSelectorWriter
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer

private[feature] trait NmiSelectorParams extends Params
  with HasFeaturesCol with HasOutputCol with HasLabelCol with DefaultParamsWritable {

  final val numTopFeatures = new IntParam(this, "numTopFeatures",
    "Number of features that selector will select.",
    ParamValidators.gtEq(1))
  setDefault(numTopFeatures, 20)

  final val delta = new DoubleParam(this, "delta",
    "The threshold of neighborhood relationship.",
    ParamValidators.gt(0)
  )

  /** Default value is the number of attributes.
    *
    */
  final val numPartitions = new IntParam(this, "numPartitions",
    "Number of partitions of algorithm.",
    ParamValidators.gtEq(1))

  final val regression = new BooleanParam(this,"regression",
  "Classification or regression")
  setDefault(regression, false)

  final val nominalIndices = new IntArrayParam(this, "nominalIndices",
  "Nominal attribute indices")

  final val sparse = new BooleanParam(this, "sparse",
  "Dataset is a sparse data.")
  setDefault(sparse, false)

  final val numBins = new IntParam(this,"numBins",
  "number of bins to estimate neighbor information.",
    ParamValidators.gtEq(1000))
  setDefault(numBins, 10000)


  // TODO param of label missing value name

}


class NmiSelector(override val uid: String)
  extends Estimator[NmiSelectorModel] with NmiSelectorParams {

  def this() = this(Identifiable.randomUID("NmiSelector"))

  def setNumTopFeatures(value: Int): this.type = set(numTopFeatures, value)

  def setDelta(value: Double): this.type = set(delta, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setNumPartitions(value: Int): this.type = set(numPartitions, value)

  def isRegression(value: Boolean): this.type = set(regression, value)

  def setNominalIndices(value: Array[Int]): this.type = set(nominalIndices, value)

  def isSparse(value: Boolean): this.type = set(sparse, value)

  def setNumBins(value: Int): this.type = set(numBins, value)

  override def fit(dataset: Dataset[_]): NmiSelectorModel = {
    transformSchema(dataset.schema)
    // if success, numAttributes well be None while attributes will contain info.
    val attributeGroup = AttributeGroup.fromStructField(dataset.schema($(featuresCol)))
    // TODO get missing value info
    val label = Attribute.fromStructField(dataset.schema($(labelCol)))
    // transform origin data into dataset can be handle by DistributeNeighborEntropy.
    val data = dataset.select(col($(labelCol)), col($(featuresCol)))

    val attr = if (attributeGroup.attributes.nonEmpty) {
      attributeGroup.attributes.get
    } else {
      val numAttr = data.first.getAs[Vector](1).size
      Array.fill[Attribute](numAttr)(NumericAttribute.defaultAttr)
    }

    val numAttributes = attr.length
    setDefault(numPartitions, numAttributes)
    val defaultNominalAttr = attr.filter(_.isNominal).map(_.index.get)
    setDefault(nominalIndices, defaultNominalAttr)

    val nominalSet: Set[Int] =
      if($(regression)) $(nominalIndices).toSet
      else $(nominalIndices).toSet + numAttributes

    val bcNominalIndices = data.sparkSession.sparkContext.broadcast(nominalSet)
    // 稀疏模式下全零列将被直接去掉，因为map阶段的输出需要存在value，reduce时才有值
    val formattedData =
      if($(sparse)) FormatConverter.convertSparse(data, numAttributes, $(numPartitions), bcNominalIndices)
      else FormatConverter.convertDens(data, numAttributes, $(numPartitions), bcNominalIndices)

    formattedData.persist(StorageLevel.MEMORY_AND_DISK)

    val attrCol = formattedData.filter(_.index != numAttributes)
    val classCol = formattedData.filter(_.index == numAttributes).first

    val neighborEntropy = new DistributeNeighborEntropy(formattedData,$(delta), $(numBins))

    val selected = new ArrayBuffer[Int]()
    // TODO implement a semi-supervised version
    // 这里可以创建一个Rdd过滤掉无类标数据
    val relevance = neighborEntropy.mutualInformation(classCol, attrCol).toMap
    val accumulateRedundancy = Array.ofDim[Double](numAttributes)
    var currentSelected = relevance.maxBy(_._2)
    selected += currentSelected._1
    while (selected.size < $(numTopFeatures)) {
      val candidateCol = attrCol.filter(col => !selected.contains(col.index))
      val previousCol = attrCol.filter(col => col.index == currentSelected._1).first()
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
    formattedData.unpersist()
    println("feature selected------>" + selected.mkString(","))
    copyValues(new NmiSelectorModel(uid, selected.toArray).setParent(this))
  }


  override def copy(extra: ParamMap): Estimator[NmiSelectorModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.checkNumericType(schema, $(labelCol))
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

object NmiSelector extends DefaultParamsReadable[NmiSelector] {
  override def load(path: String): NmiSelector = super.load(path)
}

final class NmiSelectorModel private[ml](
    override val uid: String,
    val selectedFeatures: Array[Int]) extends Model[NmiSelectorModel] with NmiSelectorParams with MLWritable {
  private val filterIndices = selectedFeatures.sorted

  override def copy(extra: ParamMap): NmiSelectorModel = {
    val copied = new NmiSelectorModel(uid, selectedFeatures)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new NmiSelectorWriter(this)

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
      Array.fill[Attribute](filterIndices.length)(NumericAttribute.defaultAttr)
    }
    val newAttrGroup = new AttributeGroup($(outputCol), selectAttr)
    newAttrGroup.toStructField
  }
}

object NmiSelectorModel extends MLReadable[NmiSelectorModel] {

  private[NmiSelectorModel] class NmiSelectorWriter(instance: NmiSelectorModel) extends MLWriter {

    private case class Data(selectedFeatures: Seq[Int])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.selectedFeatures.toSeq)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class NmiSelectorReader extends MLReader[NmiSelectorModel] {
    private val className = classOf[NmiSelectorModel].getName

    override def load(path: String): NmiSelectorModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath).select("selectedFeatures").head
      val selectedFeatures = data.getAs[Seq[Int]](0).toArray
      val model = new NmiSelectorModel(metadata.uid, selectedFeatures)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  override def read: MLReader[NmiSelectorModel] = new NmiSelectorReader

  override def load(path: String): NmiSelectorModel = super.load(path)
}