package org.apache.spark.ml.feature

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.{DenseVector, Vector, SparseVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

object FormatConverter {
  /**
    * Convert DataFrame into a column form RDD[(Int, Array[Double])]. First element of tuple is index of column,
    * second is value of origin instances in this column.
    * @param df DataFrame data.
    * @param numPartitions Number of result's partition.
    * @return Column RDD
    */
  def rotateDens(df: DataFrame, numPartitions: Int): RDD[(Int, Vector)] = {
    val numAttributes = df.first().getAs[Vector](1).size + 1
    val columnarRDD = df.rdd.mapPartitionsWithIndex{ case (pIndex, iter) =>
      val rows = iter.toArray
      println("pId: "+ pIndex +" --------------->" + numAttributes + " * " + rows.length)
      val mat = Array.ofDim[Double](numAttributes, rows.length)
      var j = 0
      for (row <- rows) {
        val dv = row.getAs[Vector](1).toDense
        val label = row.getDouble(0)
        for (i <- 0 until dv.size) mat(i)(j) = dv(i)
        mat(dv.size)(j) = label
        j += 1
      }
      val chunks = for (i <- 0 until numAttributes) yield (ColumnTransferKey(i, pIndex), mat(i)) //不使用三元组节省减小传输时间？
      chunks.toIterator
    }
    implicit def orderByColumnTransferKey[A <: ColumnTransferKey] : Ordering[A] =
      Ordering.by(k => (k.colIndex, k.partIndex))
    columnarRDD.repartitionAndSortWithinPartitions(new ColumnTransferPartitioner(numPartitions))
      .mapPartitions(iter =>
        iter.toArray.groupBy(_._1.colIndex).map(v =>(v._1, new DenseVector(v._2.flatMap(_._2)))).toIterator)
  }

  def rotateSparse(df: DataFrame, numPartitions: Int): RDD[(Int, Vector)] = {

    ???
  }

  def formatData(data: RDD[(Int, Vector)], bcNominalIndices: Broadcast[Set[Int]]): RDD[ColData] = {
    data.map { pair =>
      if(! bcNominalIndices.value.contains(pair._1)) pair._2 match {
        case v: DenseVector =>
          val values = v.values
          val valMax = values.max
          val valMin = values.min
          val max = if (valMax < 0) 0 else valMax
          val min = if (valMin > 0) 0 else valMin
          val originScale = max - min
          val newValues =
            if (max > min) values.map(v => (v - min) / originScale)
            else values
          ColData.numerical(pair._1, new DenseVector(newValues))
        case v: SparseVector =>
          val values = v.values
          val max = values.max
          val min = values.min
          val originScale = max - min
          val newValues =
            if (max > min)
              values.map(v => (v - min) / originScale)
            else
              values
          ColData.numerical(pair._1, new SparseVector(v.size, v.indices, newValues))
      } else {
        ColData.nominal(pair._1, pair._2)
      }
    }
  }

}

/**
  * Case of Column data
  * @param index index of this column
  * @param isNominal whether the attribute of this column is nominal
  * @param vector value of this column
  */
trait ColData

object ColData{
  def numerical(index: Int, vector: Vector) = NumericalColData(index, vector)
  def nominal(index: Int, vector: Vector) = NominalColData(index, vector)
}

case class NumericalColData(index: Int, vector: Vector) extends ColData

case class NominalColData(index: Int, vector: Vector) extends ColData
