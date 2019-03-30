package org.apache.spark.ml.feature

import org.apache.spark.ml.feature.LocalNeighborEntropy._
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import util.Random._
import org.apache.spark.ml.linalg.Vectors

class LocalNeighborEntropyTest extends FunSuite with BeforeAndAfterAll{
  final val FILE_PREFIX = "src/test/resources/data/"
  val num = 10000

//  test("test the precision of estimate mutual information should small enough")
//  {
//    val values = (1 to num).toArray.map(_ => util.Random.nextDouble()).map(v =>(v, v*v))
//    val (x, y) = values.unzip
//    val delta = 0.2
//    val ex = entropy(x, delta)
//    val ey = entropy(y, delta)
//    val estimateJointEntropy = pureNumJointEntropy(values, delta)
//    val estimate = ex + ey - estimateJointEntropy
//    val exact = exactNMI(values,delta)
//    println(estimate + "\n" + exact)
//    assert(math.abs(estimate - exact) < 0.0001)
//  }
//
//  test("nominal and numerical"){
//    val values = (1 to num).toArray.map(_ => (nextInt(20).toDouble,nextDouble()))
//    val (x, y) = values.unzip
//    val delta = 0.1
//    val exy = mixJointEntropy(values, delta)
//    val ex = entropy(x)
//    val ey = entropy(y, delta)
//    val mi = ex + ey - exy
//    val exact = exactNMI(values.map(p=>(p._1.toDouble, p._2)),delta)
//    println(mi + "\n" + exact)
//    assert(math.abs(mi - exact) < 0.00001)
//  }
//
//  def log2(x: Double): Double = math.log(x) / math.log(2)
//
//  test("nominal"){
//    val values = Array((1,1),(2,1),(1,2),(2,2)).map(p => (p._1.toDouble, p._2.toDouble))
//    val (x, y) = values.unzip
//    val ex = entropy(x)
//    println(ex)
//    val ey = entropy(y)
//    val exy = pureNomJointEntropy(values)
//    val mi = ex + ey - exy
//    assert(ex == -2 * 1.0/2 * log2(1.0/2))
//    val mi1 = 0
//    assert(mi == mi1)
//  }
//
//  test("ionosphere") {
//    val dataLine = scala.io.Source.fromFile(FILE_PREFIX + "ionosphere.csv").getLines()
//    val dataArray = dataLine.map(_.split(",")).toArray.tail
//    val labelString = dataArray.map(_.last)
//    val labelValue = Set(labelString: _*).zipWithIndex.toMap
//    val label = labelString.map(labelValue(_))
//    val values = dataArray.map(d=>d(6).toDouble)//.map(_=>scala.util.Random.nextDouble())
//    val first = scale(values)
//    val pairs = first.zip(label.map(_.toDouble))
//    val mi = exactNMI(first.zip(label.map(_.toDouble)), 0.1)
//    println(mi)
//
//    val ex = entropy(first, 0.1)
//
//    val colx = ColData.numerical(1,Vectors.dense(values).toSparse,values.max,values.min)
//    val sex = sparseNumericalEntropy(colx,0.1)
//    println("sex  "+sex)
//    println("ex   "+ex)
//    val ey = entropy(label.map(_.toDouble), 0.1)
//    val exy = pureNumJointEntropy(first.zip(label.map(_.toDouble)), 0.1)
//    println(ex+ey-exy)
//  }

  test("ionosphere_sparse") {
    val dataLine = scala.io.Source.fromFile(FILE_PREFIX + "ionosphere.csv").getLines()
    val dataArray = dataLine.map(_.split(",")).toArray.tail
    val labelString = dataArray.map(_.last)
    val labelValue = Set(labelString: _*).zipWithIndex.toMap
//    val v2 = labelString.map(labelValue(_)).map(_.toDouble)
    val v1 = dataArray.map(d=>d(4).toDouble)//.map(_=>scala.util.Random.nextDouble())
    val v2 = dataArray.map(d=>d(1).toDouble)//.map(_=>scala.util.Random.nextDouble())
    import scala.util.Random
    val (t1,t2) = {
      val tmp = v1.zip(v2)//.filter(p=> p._2 != 0 || p._1 != 0).sortBy(_._1)
      (tmp ++
        Array.fill(0)((Random.nextDouble(),Random.nextDouble().toInt.toDouble)) ++
        Array.fill(0)((0.0,0.0))).unzip
    }
    val before =scale(t1).zip(scale(t2))

    def generateData(v: Array[Double]) =
      ColData.numerical(1,Vectors.dense(v),v.max,v.min)
    val scol1 = generateData(t1)
    val scol2 = generateData(t2)

//    val scol11 = ColData.numerical(1,Vectors.dense(t1),t1.max,t1.min)
//    val scol21 = ColData.nominal(1,Vectors.dense(t2))
    val delta = 0.1

    val se = entropy(scol1,delta)
    val de = entropy(scol1,delta)
    println(se+"\t"+de)

    val time1 = System.currentTimeMillis
    val sje = jointEntropy(scol1,scol2,delta)
//    val dje = jointEntropy(scol11, scol21, delta)
    val time2 = System.currentTimeMillis
    val je = LocalDensNeighborEntropy.exactNJE(scale(t1).zip(scale(t2)),delta)
    val time3 = System.currentTimeMillis
//    val sje = jointEntropy(scol1,scol2,delta)
    val dje = jointEntropy(scol1, scol2, delta)
    val time4 = System.currentTimeMillis
    println("sje\t"+sje)
    println("je\t"+je)
    println("dje\t"+dje)
    println(time2-time1)
    println(time3-time2)
    println(time4-time3)



    val smi = entropy(scol1, delta) + entropy(scol2, delta) - sje
    val dmi = entropy(scol1, delta) + entropy(scol2, delta) - dje
    val e1 = LocalDensNeighborEntropy.entropy(scale(t1), delta)
    val e2 = LocalDensNeighborEntropy.entropy(scale(t2), delta)
    val mi = e1 + e2 - je
    println(smi)
    println(dmi)
    println(mi)
  }

  def scale(values: Array[Double]) = {
    val max = values.max
    val min = values.min
    val originScale = max - min
    val first =
      if (max > min)
        values.map(v => (v - min) / originScale)
      else
        values
    first
  }
}
