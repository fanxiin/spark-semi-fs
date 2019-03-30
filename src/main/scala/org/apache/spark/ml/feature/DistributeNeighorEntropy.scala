package org.apache.spark.ml.feature

import org.apache.spark.rdd.RDD

class DistributeNeighborEntropy(
    dataset: RDD[ColData],
    val delta: Double,
    numBins: Int = 10000) {
  private val sc = dataset.context
  val entropyMap: Map[Int, Double] = entropyArray(dataset).toMap

  private def entropyArray(colSet: RDD[ColData]): Array[(Int, Double)] = {
    val delta_ = delta
    val entropyRdd = colSet.map{ colData =>
      (colData.index, LocalNeighborEntropy.entropy(colData, delta_))
    }
    val result = entropyRdd.collect()
    result
  }

  /**
    * Compute joint entropy between col and each column in colSet.
    * @param col The fixed column
    * @param colSet Set of column to be compute with.
    * @return Result of entropy set.
    */
  private def jointEntropyArray(
      col: ColData,
      colSet: RDD[ColData]): Array[(Int, Double)] = {
    val delta_ = delta
    val numBins_ = numBins
    val col_ = sc.broadcast(col)
    val entropyRdd = colSet.map{ that =>
      (that.index, LocalNeighborEntropy.jointEntropy(col_.value, that, delta_, numBins_))
    }
    val result = entropyRdd.collect()
    col_.destroy()
    result
  }

  def mutualInformation(
      col: ColData,
      colSet: RDD[ColData]): Array[(Int, Double)]= {
    val jointEntropy = jointEntropyArray(col, colSet)
    val h_x = entropyMap(col.index)
    jointEntropy.map{
      case (index, h_xy) =>
        val h_y = entropyMap(index)
        (index, h_x + h_y - h_xy)
    }
  }

}

