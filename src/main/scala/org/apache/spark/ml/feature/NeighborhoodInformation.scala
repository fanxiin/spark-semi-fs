package org.apache.spark.ml.feature

import org.apache.spark.Partitioner
import org.apache.spark.ml.attribute.{AttributeGroup, NominalAttribute}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.storage.StorageLevel

class NeighborhoodInformation(
    val delta: Double,
    val nominalIndices: Set[Int]) {
  type DataFormat

  private def computeContingencyTable(
      data: RDD[DataFormat]
  )= ???
}

object NeighborhoodInformation{
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

  def rotateDFasRDD(df: DataFrame): RDD[DataFormat] = {
    val oldPartitions = df.rdd.getNumPartitions
    val numElements = df.count
    val numAttributes = df.first().getAs[Vector](1).size + 1
    val eqDistributeRDD = df.rdd.zipWithIndex().map(_.swap).partitionBy(new ExactPartition(oldPartitions, numElements))
    val columnarRDD = eqDistributeRDD.mapPartitionsWithIndex{ case (pIndex, iter) =>
      val rows = iter.toArray.map(_._2)
      val mat = Array.ofDim[Double](numAttributes, rows.length)
      var j = 0
      for (row <- rows) {
        val dv = row.getAs[Vector](1).toDense
        val label = row.getDouble(0)
        for (i <- 0 until dv.size) mat(i)(j) = dv(i)
        mat(dv.size)(j) = label
        j += 1
      }
      val chunks = for (i <- 0 until numAttributes) yield (i * oldPartitions + pIndex, mat(i))
      chunks.toIterator
    }
    val numChunks = oldPartitions * numAttributes
    columnarRDD.sortByKey()
      .partitionBy(new ColumnPartition(numAttributes, numChunks))
      .persist(StorageLevel.MEMORY_ONLY)
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
