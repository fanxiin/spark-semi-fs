package org.apache.spark.ml.feature

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, MLWritable, MLWriter}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.functions._

private[feature] trait SemiSelectorParams extends Params
  with HasFeaturesCol with HasOutputCol with HasLabelCol with DefaultParamsWritable{

}


class SemiSelector(override val uid: String)
  extends Estimator[SemiSelectorModel] with SemiSelectorParams {
  override def fit(dataset: Dataset[_]): SemiSelectorModel = {
    transformSchema(dataset.schema)
    val input = dataset.select(col($(labelCol)).cast(DoubleType), col($(featuresCol))).rdd.map {
      case Row(label: Double, features: Vector) =>
        LabeledPoint(label, features)
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