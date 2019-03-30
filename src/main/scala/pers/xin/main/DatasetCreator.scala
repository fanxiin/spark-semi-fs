package pers.xin.main

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class DatasetCreator(
    labelName: String,
    missingString: String,
    val trainPart: Double,
    val labelRatio: Double) {

  def tag(data: DataFrame): DataFrame = {
    val ratio_ = labelRatio
    val missingString_ = missingString
    val randomTag = udf { label: String =>
      if (scala.util.Random.nextDouble() < ratio_)
        label
      else
        missingString_
    }
    // 一定要在之后持久化，否则从新计算随机数会不同
    data.withColumn(labelName, randomTag(col(labelName)))
  }

  def create(data: DataFrame): (DataFrame, DataFrame) = {
    val Array(train, test) = data.randomSplit(Array(trainPart, 1 - trainPart))
    (tag(train), test)
  }
}

object DataLoader{
  def load(dataPath: String, schemaPath: String): Unit ={

  }
}
