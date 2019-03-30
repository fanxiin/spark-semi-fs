package pers.xin.main

import org.apache.spark.SparkContext
import org.apache.spark.ml.attribute.{Attribute, NominalAttribute}
import org.apache.spark.ml.feature.{RFormula, SemiHelper, SemiSelector}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class DatasetCreatorTest extends FunSuite with BeforeAndAfterAll{
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

  test("create dataset"){
    val origin = spark.read.format("csv")
      .option("inferSchema", true)
      .option("header", true)
      .load(FILE_PREFIX + "ionosphere.csv")

    val createDataset = new DatasetCreator("v34","__miss__", 0.7,0.3)
    val (df, test) = createDataset.create(origin)
    df.persist()
    df.show()


    val rf = new RFormula()
      .setFormula("v34 ~ .")
      .fit(df)
    val data = rf.transform(df).select("label","features")
    data.show()
    val labelAttr = Attribute.fromStructField(data.schema("label"))
    val missIndex = labelAttr.asInstanceOf[NominalAttribute].indexOf("__miss__")
    data.where(s"label != ${missIndex}").show()
    data.show()
    val selector = new SemiSelector()
      .setNumTopFeatures(32)
      .setDelta(0.1)
      .setNumPartitions(10)
      .isSparse(true)
      .setOutputCol("my_features")
      .setNominalIndices(Array(0))
    val model = selector.fit(data)
    val result = model.transform(data)
    result.show()
  }
  test("create dataset1"){
    val df = spark.read.format("csv")
      .option("inferSchema", true)
      .option("header", true)
      .load(FILE_PREFIX + "ionosphere.csv")

    val rf = new RFormula()
      .setFormula("v34 ~ .")
      .fit(df)
    val data1 = rf.transform(df).select("label","features")
    val data = SemiHelper.tag(data1,"label",0.5,SemiSelector.MISS_VALUE).persist()
    data.show()
    val selector = new SemiSelector()
      .setNumTopFeatures(32)
      .setDelta(0.1)
      .setNumPartitions(10)
      .isSparse(true)
      .setNominalIndices(Array(0))
    val model = selector.fit(data)
    val result = model.transform(data)
    result.show()
  }
}
