package org.apache.spark.ml.feature

import org.apache.spark.Partitioner
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.attribute.{Attribute, NominalAttribute}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import scala.collection.mutable.ArrayBuffer

object FormatConverter {
  /**
    * Convert DataFrame into a column form RDD[(Int, Array[Double])]. First element of tuple is index of column,
    * second is value of origin instances in this column.
    * @param df DataFrame data.
    * @param numPartitions Number of result's partition.
    * @return Column RDD
    */
  def rotateDens(df: DataFrame, numAttributes: Int, numPartitions: Int): RDD[(Int, Vector)] = {
//    val numAttributes = df.first().getAs[Vector](1).size + 1
    val columnarRDD = df.rdd.mapPartitionsWithIndex{ case (pIndex, iter) =>
      val rows = iter.toArray
      println("pId: "+ pIndex +" --------------->" + numAttributes + " * " + rows.length)
      val mat = Array.ofDim[Double](numAttributes + 1, rows.length)
      var j = 0
      for (row <- rows) {
        val dv = row.getAs[Vector](1).toDense
        val label = row.getDouble(0)
        for (i <- 0 until dv.size) mat(i)(j) = dv(i)
        mat(dv.size)(j) = label
        j += 1
      }
      val chunks = for (i <- 0 to numAttributes) yield (ColumnTransferKey(i, pIndex), mat(i)) //不使用三元组节省减小传输时间？
      chunks.toIterator
    }
    implicit def orderByColumnTransferKey[A <: ColumnTransferKey] : Ordering[A] =
      Ordering.by(k => (k.colIndex, k.partIndex))
    columnarRDD.repartitionAndSortWithinPartitions(new ColumnTransferPartitioner(numPartitions))
      .mapPartitions(iter =>
        iter.toArray.groupBy(_._1.colIndex).map(v =>(v._1, new DenseVector(v._2.flatMap(_._2)))).toIterator)
  }

  def rotateSparse(df: DataFrame, numAttributes: Int, numPartitions: Int): RDD[(Int, Vector)] = {
    val size = df.count.toInt
    val columnarRDD = df.rdd.zipWithIndex.flatMap { case (row, rIndex) =>
      val sv = row.getAs[Vector](1).toSparse
      val label = row.getDouble(0)
      val indices = sv.indices
      val values = sv.values
      val buffer = new ArrayBuffer[(ColumnTransferKey, Double)](sv.indices.length)
      for (i <- 0 until indices.length)
        buffer += ((new ColumnTransferKey(sv.indices(i), rIndex ), sv.values(i)))
      buffer += ((new ColumnTransferKey(numAttributes, rIndex ), label))
      buffer.toIterator
    }
    implicit def orderByColumnTransferKey[A <: ColumnTransferKey] : Ordering[A] =
      Ordering.by(k => (k.colIndex, k.partIndex))

    def mkSparse(info: Array[(ColumnTransferKey, Double)]): SparseVector = {
      val (key, values) = info.unzip
      val indices = key.map(_.partIndex.toInt)
      new SparseVector(size, indices, values)
    }
    columnarRDD.repartitionAndSortWithinPartitions(new ColumnTransferPartitioner(numPartitions))
      .mapPartitions(iter =>
        iter.toArray.groupBy(_._1.colIndex)
          .map(v =>(v._1, mkSparse(v._2))).toIterator)
  }

  def formatData(data: RDD[(Int, Vector)], bcNominalIndices: Broadcast[Set[Int]]): RDD[ColData] = {
    data.map { pair =>
      if(! bcNominalIndices.value.contains(pair._1)) pair._2 match {
        case v: DenseVector =>
          val values = v.values
          val max = values.max
          val min = values.min
          ColData.numerical(pair._1, pair._2, max, min)
        case v: SparseVector =>
          val values = v.values
          val valMax = values.max
          val valMin = values.min
          val max = if (valMax < 0) 0 else valMax
          val min = if (valMin > 0) 0 else valMin
          ColData.numerical(pair._1, pair._2, max, min)
      } else {
        ColData.nominal(pair._1, pair._2)
      }
    }
  }

  def convertDens(
      df: DataFrame,
      numAttributes: Int,
      numPartitions: Int,
      bcNominalIndices: Broadcast[Set[Int]]): RDD[ColData] = {
    formatData(rotateDens(df, numAttributes, numPartitions), bcNominalIndices)
  }

  def convertSparse(
      df: DataFrame,
      numAttributes: Int,
      numPartitions: Int,
      bcNominalIndices: Broadcast[Set[Int]]): RDD[ColData] = {
    formatData(rotateSparse(df, numAttributes, numPartitions), bcNominalIndices)
  }
}

object SemiHelper{
  def tag(
      data: DataFrame,
      label: String,
      ratio: Double,
      missingName: String): DataFrame = {

    val originAttr = Attribute.fromStructField(data.schema(label)).asInstanceOf[NominalAttribute]
    val name = originAttr.name
    val index = originAttr.index
    val isOrdinal = originAttr.isOrdinal
    val numValues = originAttr.numValues.map(_+1)
    val values = originAttr.values.map(_ :+ missingName)
    val newAttr = new NominalAttribute(name, index, isOrdinal, numValues, values)
    val missingIndex = values.get.length - 1
    val randomTag = udf { label: Double =>
      if (scala.util.Random.nextDouble() < ratio)
        label
      else
        missingIndex
    }
    // 一定要在之后持久化，否则从新计算随机数会不同
    data.withColumn(label, randomTag(col(label)), newAttr.toMetadata())
  }
}

case class ColumnTransferKey(colIndex: Int, partIndex: Long)

class ColumnTransferPartitioner(override val numPartitions: Int) extends Partitioner{
  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[ColumnTransferKey]
    k.colIndex % numPartitions
  }
}
//
//class ExactPartitioner(override val numPartitions: Int) extends Partitioner {
//  override def getPartition(key: Any): Int = {
//    val k = key.asInstanceOf[Long]
//    k % numPartitions
//  }
//}

trait ColData {
  def index: Int
  def vector: Vector
}

object ColData{
  def numerical(index: Int, vector: Vector, max: Double, min: Double) =
    NumericalColData(index, vector, max, min)
  def nominal(index: Int, vector: Vector) = NominalColData(index, vector)
}

case class NumericalColData(
    override val index: Int,
    override val vector: Vector,
    max: Double,
    min: Double) extends ColData

case class NominalColData(override val index: Int, override val vector: Vector) extends ColData
