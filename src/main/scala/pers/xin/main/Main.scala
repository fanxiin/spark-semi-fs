package pers.xin.main

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{SemiSelector, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object Main {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("feature selection")
      .getOrCreate()
    val df = spark.read.format("libsvm")
      .option("inferSchema", true)
      .load(args(0))

    val repartDF = df.repartition(2000).persist()

    val model = new SemiSelector()
      .setDelta(args(1).toDouble)
      .setNumTopFeatures(args(2).toInt)
      .setOutputCol("selected").fit(df)

    val result = model.transform(df)

    result.show()
  }

}
