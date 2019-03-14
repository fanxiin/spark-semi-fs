package org.apache.spark.ml.feature

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.VectorUDT

private[feature] trait SemiSelectorParams extends Params
  with HasFeaturesCol with HasOutputCol with HasLabelCol with DefaultParamsWritable{

  final val numTopFeatures = new IntParam(this, "numTopFeatures",
  "Number of features that selector will select.",
    ParamValidators.gtEq(1))
  setDefault(numTopFeatures, 20)

  final val delta = new DoubleParam(this, "delta",
    "The threshold of neighborhood relationship.",
    ParamValidators.gt(0)
  )

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
    val data = dataset.select(col($(labelCol)),col($(featuresCol)))

    val rotatedData = NeighborEntropyHelper.rotateDFasRDD(data, 10)

    // TODO numerical column should be locate between 0 and 1

    val attrCol = rotatedData.filter(_._1 != numAttributes)
    val classCol = rotatedData.filter(_._1 == numAttributes).first
    val neighborEntropy = new DistributeNeighborEntropy(
      $(delta),
      rotatedData,
      nominalIndices
    )
    val selected = collection.mutable.Set[Int]()
    val relevance = neighborEntropy.mutualInformation(classCol, attrCol).toMap
    val accumulateRedundancy = Array.ofDim[Double](numAttributes)
    var currentSelected = relevance.maxBy(_._2)
    selected += currentSelected._1
    while (selected.size < $(numTopFeatures)) {
      val candidateCol = attrCol.filter(col => ! selected.contains(col._1))
      val previousCol = attrCol.filter(_._1 == currentSelected._1).first()
      val mis = neighborEntropy.mutualInformation(previousCol, candidateCol)
      mis.foreach {case (index, mi) => accumulateRedundancy(index) += mi}
      val selectedCounts = selected.size
      val measures = mis.map{
        case(index, _) =>
          val sig = relevance(index) - accumulateRedundancy(index) / selectedCounts
          (index, sig)
      }
      currentSelected = measures.maxBy(_._2)
      selected += currentSelected._1
    }
    new SemiSelectorModel(uid, selected.toArray)
  }


  override def copy(extra: ParamMap): Estimator[SemiSelectorModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.checkNumericType(schema, $(labelCol))
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

final class SemiSelectorModel private[ml] (
    override val uid: String,
    private val selectedFeatures: Array[Int]) extends Model[SemiSelectorModel] with SemiSelectorParams with MLWritable {
  private val filterIndices = selectedFeatures.sorted

  override def copy(extra: ParamMap): SemiSelectorModel = {
    val copied = new SemiSelectorModel(uid, selectedFeatures)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = ???

  override def transform(dataset: Dataset[_]): DataFrame = {
    val slicer = new VectorSlicer()
        .setInputCol($(featuresCol))
        .setOutputCol($(outputCol))
        .setIndices(filterIndices)
    slicer.transform(dataset)
  }

  override def transformSchema(schema: StructType): StructType = ???
}