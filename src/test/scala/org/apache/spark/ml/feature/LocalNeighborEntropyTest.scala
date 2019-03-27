package org.apache.spark.ml.feature

import org.apache.spark.ml.feature.LocalNeighborEntropy._
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import util.Random._

class LocalNeighborEntropyTest extends FunSuite with BeforeAndAfterAll{
  final val FILE_PREFIX = "src/test/resources/data/"
  val num = 10000

  test("test the precision of estimate mutual information should small enough")
  {
    val values = (1 to num).toArray.map(_ => util.Random.nextDouble()).map(v =>(v, v*v))
    val (x, y) = values.unzip
    val delta = 0.2
    val ex = entropy(x, delta)
    val ey = entropy(y, delta)
    val estimateJointEntropy = pureNumJointEntropy(values, delta)
    val estimate = ex + ey - estimateJointEntropy
    val exact = exactNMI(values,delta)
    println(estimate + "\n" + exact)
    assert(math.abs(estimate - exact) < 0.0001)
  }

  test("nominal and numerical"){
    val values = (1 to num).toArray.map(_ => (nextInt(20).toDouble,nextDouble()))
    val (x, y) = values.unzip
    val delta = 0.1
    val exy = mixJointEntropy(values, delta)
    val ex = entropy(x)
    val ey = entropy(y, delta)
    val mi = ex + ey - exy
    val exact = exactNMI(values.map(p=>(p._1.toDouble, p._2)),delta)
    println(mi + "\n" + exact)
    assert(math.abs(mi - exact) < 0.00001)
  }

  def log2(x: Double): Double = math.log(x) / math.log(2)

  test("nominal"){
    val values = Array((1,1),(2,1),(1,2),(2,2)).map(p => (p._1.toDouble, p._2.toDouble))
    val (x, y) = values.unzip
    val ex = entropy(x)
    println(ex)
    val ey = entropy(y)
    val exy = pureNomJointEntropy(values)
    val mi = ex + ey - exy
    assert(ex == -2 * 1.0/2 * log2(1.0/2))
    val mi1 = 0
    assert(mi == mi1)
  }

  test("ionosphere") {
    val dataLine = scala.io.Source.fromFile(FILE_PREFIX + "ionosphere.csv").getLines()
    val dataArray = dataLine.map(_.split(",")).toArray.tail
    val labelString = dataArray.map(_.last)
    val labelValue = Set(labelString: _*).zipWithIndex.toMap
    val label = labelString.map(labelValue(_))
    val values = dataArray.map(d=>d(2).toDouble)

    val max = values.max
    val min = values.min
    val originScale = max - min
    val first =
      if (max > min)
        values.map(v => (v - min) / originScale)
      else
        values


    val pairs = first.zip(label.map(_.toDouble))

    val mi = exactNMI(first.zip(label.map(_.toDouble)), 0.1)
    println(mi)

    val ex = entropy(first, 0.1)
    val ey = entropy(label.map(_.toDouble), 0.1)
    val exy = pureNumJointEntropy(first.zip(label.map(_.toDouble)), 0.1)
    println(ex+ey-exy)
  }
}
