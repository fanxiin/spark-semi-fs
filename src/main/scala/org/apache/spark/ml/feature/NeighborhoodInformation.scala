package org.apache.spark.ml.feature

import org.apache.spark.Partitioner
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.storage.StorageLevel

class NeighborhoodInformation(
    val delta: Double,
    val dataset: RDD[(Int, Array[Double])],
    val labels: Array[Byte],
    val numLabels: Int,
    val nominalIndices: Set[Int]) {
  val sc = dataset.context
  val getRelevance: Array[Double] = {
    val bLabels = sc.broadcast(labels)
    val bNominalIndices = sc.broadcast(nominalIndices)
    val delta_ = delta
    dataset.map{
      case (colIndex, col) if bNominalIndices.value.contains(colIndex) =>

    }
    ???
  }

}

object NeighborhoodInformationHelper{
  type DataFormat = (Int, Array[Double])
  def rotateRDD(data: RDD[LabeledPoint], numPartitions: Int = 0): RDD[DataFormat] = {
    val oldPartitions = data.getNumPartitions
    val numElements = data.count()
    val numAttributes = data.first.features.size + 1
    val eqDistributedData = data.zipWithIndex.map(_.swap).partitionBy(new ExactPartition(oldPartitions, numElements))
    val columnarData = eqDistributedData.mapPartitionsWithIndex { case (pIndex, iter) =>
      val data = iter.toArray.map(_._2)
      val mat = Array.ofDim[Double](numAttributes, data.length)
      var j = 0
      for (lp <- data){
        val dv = lp.features.toDense
        for(i <- 0 until dv.size) mat(i)(j) = dv(i)
        mat(dv.size)(j) = lp.label
        j += 1
      }
      val chunks = for (i <- 0 until numAttributes) yield (i * oldPartitions + pIndex, mat(i))
      chunks.toIterator
    }
    val numChunks = oldPartitions * numAttributes
    val denseData = columnarData.sortByKey()
      .partitionBy(new ColumnPartition(numAttributes, numChunks))
      .persist(StorageLevel.MEMORY_ONLY)
    denseData
  }

  /**
    * Convert DataFrame into a column form RDD[(Int, Array[Double])]. First element of tuple is index of column,
    * second is value of origin instances in this column.
    * @param df DataFrame data.
    * @param numPartitions Number of result's partition.
    * @return Column RDD
    */
  def rotateDFasRDD(df: DataFrame, numPartitions: Int): RDD[(Int, Array[Double])] = {
    val numAttributes = df.first().getAs[Vector](1).size + 1
    val columnarRDD = df.rdd.mapPartitionsWithIndex{ case (pIndex, iter) =>
      val rows = iter.toArray
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
        iter.toArray.groupBy(_._1.colIndex).map(v =>(v._1, v._2.flatMap(_._2))).toIterator)
      .persist(StorageLevel.MEMORY_ONLY)
  }
}

case class ColumnTransferKey(colIndex: Int, partIndex: Int)

class ColumnTransferPartitioner(override val numPartitions: Int) extends Partitioner{
  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[ColumnTransferKey]
    k.colIndex % numPartitions
  }
}

class ExactPartition(override val numPartitions: Int, numElements: Long)
  extends Partitioner {
  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[Long]
    (k * numPartitions / numElements).toInt
  }
}

class ColumnPartition(override val numPartitions: Int, numElements: Long)
  extends Partitioner {
  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[Int]
    (k * numPartitions / numElements).toInt
  }
}
