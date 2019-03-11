package org.apache.spark.ml.feature

import org.apache.spark.Partitioner
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.storage.StorageLevel
import breeze.linalg.{DenseMatrix => BDM}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class NeighborhoodInformation(
    val delta: Double,
    dataset: RDD[(Int, Array[Double])],
    val bLabels: Broadcast[Array[Byte]],
    val numLabels: Int,
    val nominalIndices: Broadcast[Set[Int]]) {
  type DataFormat

  private def computeContingencyTable(
  )= ???

  private def mutualInformation(
      iterA: Iterator[Double],
      iterB: Iterator[Double]): Double = {

    ???
  }

  val attrsNumValues = {
    val counts = dataset.filter(pair => nominalIndices.value.contains(pair._1))
      .map(pair => (pair._1, (pair._2.max + 1).asInstanceOf[Int]))
    dataset.context.broadcast(counts.collect().toMap)
  }

  val relevance = {
    val numClass = numLabels
    val delat_ = delta
    dataset.map{
      case (col, values) if nominalIndices.value.contains(col) =>
        val labels = bLabels.value
        val numX = attrsNumValues.value.getOrElse(col, throw new Exception("require nominal attribute"))
//        val dm = Array.ofDim(numX,numClass)
        val contingency = new BDM(numX, numClass, new Array[Double](numX * numClass))
        values.zip(labels).foreach {case (x, c) =>
            contingency(x.asInstanceOf[Int],c) += 1
        }

      case (col, values) if !nominalIndices.value.contains(col) =>
        val labels = bLabels.value
        val numX = attrsNumValues.value.getOrElse(col, throw new Exception("require nominal attribute"))
        val x = values.zip(labels).foldLeft(mutable.Map.empty[Byte, ArrayBuffer[Double]]){
          case (m, c) =>
            val buf = m.getOrElseUpdate(c._2, new ArrayBuffer[Double])
            buf += c._1
            m
        }
        x.flatMap { case (_, v) =>
            val sortedV = v.sorted
            var lower = 0
            var upper = sortedV.head + delat_
            var current = sortedV.head
            var i = 0
            while (sortedV(i) < upper) {
              i += 1
            }
            var counter = i
            for (i <- counter until sortedV.size)
              yield {
                val n_upper = sortedV(i) + delat_
              }
            ???

        }

    }
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

  def rotateDFasRDD(df: DataFrame, numPartitions: Int): RDD[(Int, Array[Double])] = {
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
      val chunks = for (i <- 0 until numAttributes) yield (ColumnTransferKey(i, pIndex), mat(i)) //不使用三元组节省减小传输时间？
      chunks.toIterator
    }
    implicit def orderByColumnTransferKey[A <: ColumnTransferKey] : Ordering[A] =
      Ordering.by(k => (k.colIndex, k.partIndex))
    columnarRDD.repartitionAndSortWithinPartitions(new ColumnTransferPartitioner(numPartitions))
      .map(pair => (pair._1.colIndex, pair._2))
      .mapPartitions(iter => {
        iter.toArray.groupBy(_._1).map(pair => (pair._1,pair._2.map(_._2).flatten)).toIterator
      }).persist(StorageLevel.MEMORY_ONLY)
  }
}

case class ColumnTransferKey(colIndex: Int, partIndex: Int)

class ColumnTransferPartitioner(override val numPartitions: Int) extends Partitioner{
  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[ColumnTransferKey]
    k.colIndex / numPartitions
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
