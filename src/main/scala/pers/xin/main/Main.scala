package pers.xin.main

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{SemiSelector, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

object Main {
  final val FILE_PREFIX = "src/test/resources/data/"
  final val CLEAN_SUFFIX = "_CLEAN"
  final val MISSING = "__MISSING__"
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("feature selection")
      .getOrCreate()
    val df = spark.read.format("csv")
      .option("header", true)
      .option("inferSchema", true)
      .load(FILE_PREFIX + "test_colon_s3.csv")
    val cleanedDf = cleanLabel(df,df.columns.head).persist(StorageLevel.MEMORY_ONLY)

    val cols = cleanedDf.columns
    val labelCol: String = cols.head
    val attrIndices: Array[String] = cols.drop(1)

    val stringIndexer = new StringIndexer()
      .setInputCol(labelCol)
      .setOutputCol("label")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(attrIndices)
      .setOutputCol("features")

    val selector = new SemiSelector()
      .setDelta(0.15)
      .setNumTopFeatures(20)
      .setFeaturesCol("features")
      .setLabelCol("label")

    val pipeline = new Pipeline()
      .setStages(Array(stringIndexer, vectorAssembler, selector))

    val model = pipeline.fit(cleanedDf)

    val result = model.transform(cleanedDf)

    result.show()
    val x = result.first()
  }

  def cleanLabel(df: DataFrame, labelColumn: String): DataFrame = {
    df.withColumn(labelColumn, when(col(labelColumn).isNull, lit(MISSING)).otherwise(col(labelColumn)))
  }
}
