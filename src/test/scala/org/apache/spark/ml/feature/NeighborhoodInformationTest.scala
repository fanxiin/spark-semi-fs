package org.apache.spark.ml.feature

import org.apache.spark.SparkContext
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class NeighborhoodInformationTest extends FunSuite with BeforeAndAfterAll {
  var spark: SparkSession = _
  var sc: SparkContext = _
  final val FILE_PREFIX = "src/test/resources/data/"
  final val CLEAN_SUFFIX = "_CLEAN"
  final val MISSING = "__MISSING__"

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder
        .master("local[2]")
        .appName("MLTest")
        .getOrCreate()
    sc = spark.sparkContext
  }

  test("intgers"){
    val df = spark.read.parquet(FILE_PREFIX + "simple-ml-integers")
    val rf = new RFormula()
      .setFormula("int3 ~ int1 + int2 + int1:int2")
      .fit(df)
    val cleanedDf = rf.transform(df).select("label", "features")
    cleanedDf.show()
    val test = spark.range(10,20).toDF().withColumn("x",col("id") + 10)
      .withColumn("l",rand()>0.5)
    test.show()
    val rf1 = new RFormula().setFormula("l ~ x + id + x : id").fit(test)
    val cleanedTest = rf1.transform(test).select("label", "features").repartition(4)
    val rot = NeighborhoodInformation.rotateDFasRDD(cleanedTest,2)
//    rot.foreach{case (i, array) => println(i+" "+array.mkString("[",",","]"))}
    rot.collectPartitions().foreach(a=>{
//      a.foreach(pair=>println(pair._1 +"\t"+ pair._2.mkString(",")))
      a.foreach{case (c, v) => println(c+"\t"+v.mkString(",")) }
      println()
    })
  }

  ignore("test"){
    val df = spark.read.format("csv")
      .option("header", true)
      .option("inferSchema", true)
      .load(FILE_PREFIX + "test_colon_s3.csv")
    val cols = df.columns
    val labelCol = cols.head
    val cleanedDf = df.withColumn(
      labelCol + CLEAN_SUFFIX,
      when(col(labelCol).isNull, lit(MISSING)).otherwise(col(labelCol)))
    val labelIndexer = new StringIndexer()
      .setInputCol(labelCol + CLEAN_SUFFIX)
      .setOutputCol("label").fit(cleanedDf)
    val labelConverted = labelIndexer.transform(cleanedDf)
    val assembler = new VectorAssembler()
      .setInputCols(cols.drop(1))
      .setOutputCol("features")
    val processedData = assembler.transform(labelConverted).select("features", "label")
    processedData.show()
//    val attr = Attribute.fromStructField(processedData.schema("label"))
//    println(attr)
    val attrGroup = AttributeGroup.fromStructField(processedData.schema("features"))
    println(attrGroup)

  }

  override protected def afterAll(): Unit = {
    if (spark != null) spark.stop()
    super.afterAll()
  }
}
