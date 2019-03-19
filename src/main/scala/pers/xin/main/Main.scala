package pers.xin.main

import org.apache.spark.ml.feature.{SemiSelector, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Main {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("feature selection")
      .getOrCreate()

    val df = args(1) match {
      case "libsvm" => spark.read.format("libsvm").load(args(0))
      case "parquet" => spark.read.parquet(args(0))
      case _ => throw new IllegalArgumentException(s"bat format ${args(1)}")
    }

    val selector = new SemiSelector()
      .setDelta(args(2).toDouble)
      .setNumTopFeatures(args(3).toInt)
      .setOutputCol("selected")

    val model = if (args.length == 4) selector.setNumPartitions(args(4).toInt).fit(df)
      else selector.fit(df)

    val result = model.transform(df)

    result.show()
    println("---" + result.first())
  }

}
