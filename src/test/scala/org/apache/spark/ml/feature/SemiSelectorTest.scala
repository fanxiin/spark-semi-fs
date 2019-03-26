package org.apache.spark.ml.feature

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class SemiSelectorTest extends FunSuite with BeforeAndAfterAll {
  var spark: SparkSession = _
  var sc: SparkContext = _
  final val FILE_PREFIX = "src/test/resources/data/"

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder
      .master("local[2]")
      .appName("MLTest")
      .getOrCreate()
    sc = spark.sparkContext
  }

  test("ionosphere"){
    val df = spark.read.format("csv")
      .option("inferSchema", true)
      .option("header", true)
      .load(FILE_PREFIX + "ionosphere.csv")

    val rf = new RFormula()
      .setFormula("v34 ~ .")
      .fit(df)
    val data = rf.transform(df).select("label","features")
    data.show()
    val selector = new SemiSelector()
      .setNumTopFeatures(10)
      .setDelta(0.15)
      .setNumPartitions(10)
    val model = selector.fit(data)
    val result = model.transform(data)
    result.show()
  }

  test("vehicle"){
    val df = spark.read.format("csv")
      .option("inferSchema", true)
      .option("header", true)
      .load(FILE_PREFIX + "vehicle.csv")

    val rf = new RFormula()
      .setFormula("v18 ~ .")
      .fit(df)
    val data = rf.transform(df).select("label","features")
    data.show()
    val selector = new SemiSelector()
      .setNumTopFeatures(10)
      .setDelta(0.15)
    val model = selector.fit(data)
    val result = model.transform(data)
    result.show()
  }
}
