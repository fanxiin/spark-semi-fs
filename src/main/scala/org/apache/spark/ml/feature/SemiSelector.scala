package org.apache.spark.ml.feature

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, MLWritable, MLWriter}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.functions._
import breeze.linalg.{DenseVector => BDV}

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
  override def fit(dataset: Dataset[_]): SemiSelectorModel = {
    transformSchema(dataset.schema)
    val attributeGroup = AttributeGroup.fromStructField(dataset.schema($(featuresCol)))
    val numAttributes = attributeGroup.numAttributes.get
    val attr = attributeGroup.attributes.get.zipWithIndex
    val nominalIndices = attr.filter(_._1.isNominal).map(_._2).toSet
    val data = dataset.select(col($(labelCol)),col($(featuresCol)))
    val rotatedData = NeighborEntropyHelper.rotateDFasRDD(data, 10)
    val neighborEntropy = new DistributeNeighborEntropy(
      $(delta),
      rotatedData,
      numAttributes,
      nominalIndices
    )
    val selected = collection.mutable.Set[Int]()
    val relevance = neighborEntropy.relevanceMap
    val accumulateRedundancy = Array.ofDim[Double](numAttributes)
    var currentSelected = relevance.maxBy(_._2)._1
    selected += currentSelected
    while (selected.size < $(numTopFeatures)) {
      val candidateCol = rotatedData.filter()
      val mis = neighborEntropy.mutualInformation(currentSelected)
      mis.foreach {case (index, mi) => accumulateRedundancy(index) += mi}
      val selectedCounts = selected.size
      val sig
    }


    ???
  }


  override def copy(extra: ParamMap): Estimator[SemiSelectorModel] = ???

  override def transformSchema(schema: StructType): StructType = ???
}

final class SemiSelectorModel private[ml] (
    override val uid: String,
    private val semiSelectorModel: SemiSelectorModel) extends Model[SemiSelectorModel] with SemiSelectorParams with MLWritable {
  override def copy(extra: ParamMap): SemiSelectorModel = ???

  override def write: MLWriter = ???

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def transformSchema(schema: StructType): StructType = ???
}