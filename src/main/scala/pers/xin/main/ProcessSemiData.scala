package pers.xin.main

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.feature.SemiSelector
import org.apache.spark.sql.SparkSession

object ProcessSemiData {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("feature selection")
      .getOrCreate()

    val dataPath = args(0)
    val outputFolder = dataPath.substring(0,dataPath.lastIndexOf("/"))
    val dataName = dataPath.substring(dataPath.lastIndexOf("/") + 1)
    val dataOutputPath = new Path(outputFolder,"selected/"+dataName+"/data").toString
    val modelOutputPath = new Path(outputFolder,"selected/"+dataName+"/model").toString

    val df = args(1) match {
      case "libsvm" => spark.read.format("libsvm").load(dataPath)
      case "parquet" => spark.read.parquet(dataPath)
      case _ => throw new IllegalArgumentException(s"bat format ${args(1)}")
    }

    val selector = new SemiSelector()
      .setDelta(args(2).toDouble)
      .setNumTopFeatures(args(3).toInt)
      .setOutputCol("selected")

    val model = if (args.length == 5) selector.setNumPartitions(args(4).toInt).fit(df)
    else if (args.length == 6) selector.setNumPartitions(args(4).toInt).isSparse(true).fit(df)
    else selector.fit(df)

    val result = model.transform(df)
    model.write.save(modelOutputPath)
    result.write.parquet(dataOutputPath)
  }

}
