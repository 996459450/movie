package com.fx.Movie

import scala.collection.Seq
import scala.io.Source

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.rdd.RDD

    // MovieLensALS
object MovieLensALS {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)
    
    val sparkconf = new SparkConf().setAppName("MovieLensALS").setMaster("local[2]")
    
    val sc = new SparkContext(sparkconf)
    
    //装载用户评分，该评分由评分器生成
    val myRatings = loadRating(args(1))
  
    val myRatingsRDD = sc.parallelize(myRatings, 1)
    
    //样本数据目录
    val movielensHomeDir = args(0)
    
    val ratings = sc.textFile(movielensHomeDir+"/ratings.dat").map { 
      line => 
        val fields = line.split("::")
        //format: (timestamp % 10 ,Rating (userId ,movieId ,rating))
        
        (fields(3).toLong % 10 ,Rating(fields(0).toInt,fields(1).toInt,fields(2).toDouble))
        }
    
    //装载电影目录对应表（电影ID -> 电影标题）
    val movies = sc.textFile(movielensHomeDir+"/movies.dat").map { 
      line => 
        val fields = line.split("::")
    //format : (movieId .movieName)
        (fields(0).toInt,fields(1))
    }.collect().toMap
    //统计有用户数量和电影数量以及用户对电影评分数目
    val numRatings = ratings.count();
    val numUsers = ratings.map(_._2.user).distinct().count();
    val numMivies = ratings.map(_._2.product).distinct().count()
    println("GOT: " +numRatings +" ratings " + numUsers +" users " + numMivies+"Movies")
    //将样本评分表一Key值切分成三个部分
//    vali
    val numPartitings = 4
    val training = ratings.filter(x => x._1 < 6).values.union(myRatingsRDD).repartition(numPartitings).persist()
  
    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8).
    values.repartition(numPartitings).persist()
    
    val test = ratings.filter(x => x._1 >= 8).values.persist()
    
    val numTraining = training.count()
    val numValidation = validation.count()
    val numTest = test.count()
    
    println("Training: " + numTraining + " validation: " + numValidation + " test: " + numTest)
  
    //训练不同参数下的模型，并在校验集中验证
    val ranks = List(8,12)
    val lambdas = List(0.1,10)
    
    val numIters = List(10,20)
    var bestModel : Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    
    var bestRank = 0
    var bestLamdba = -1.0
    var bestNumIter = -1
    
    for(rank <- ranks ; lambda <- lambdas ;numIter <- numIters){
     val model = ALS.train(training, rank, numIter,lambda)
     val validationRmse = computeRmse(model, validation, numValidation)
      
     println("RMSE(validation) = " + validationRmse + " for the model trained with rank = "

        + rank + ",lambda = " + lambda + ",and numIter = " + numIter + ".")
      
     if(validationRmse < validationRmse){
       bestModel = Some(model)
       bestValidationRmse = validationRmse
       bestRank = rank
       bestLamdba = lambda
       bestNumIter = numIter
       }  
    }

    //用最佳模型预测测试集的评分，并计算和实际评分之间的均方跟误差
    val testRmse = computeRmse(bestModel.get, test, numTest)
    
    println("The best model was trained with rank = " + 
        bestRank + " and lambda = " + bestLamdba

      + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")

     val meanRating = training.union(validation).map {_.rating }.mean()
     val baselineRmse = math.sqrt(test.map { x => 
       (meanRating -x.rating).*(meanRating- x.rating) }.reduce(_+_)/numTest)
    
      val improvement = (baselineRmse - testRmse)/baselineRmse* 100
      
    println("The best model improves the baseline by " 
        + "%1.2f".format(improvement) + "%.")

     //推荐前十部最感兴趣的电影，注意要剔除已评分的电影
     
     val myRatingMoviesIds = myRatings.map { _.product }.toSet
     val candidations = sc.parallelize(movies.keys.filter 
         { !myRatingMoviesIds.contains(_)}.toSeq)

     val recommendations = bestModel.get.predict(candidations.map { 
       (0,_) }).collect().sortBy { -_.rating }.take(10)
     
     var i =1;
     println("Movies recommended for you:")
     recommendations.foreach { r => println(
         "%2d".format(i)+":"+movies(r.product)) 
         
         i+=1;
     }
     sc.stop()     
  }
  
  
  /**
   * 校验集预测数据和实际数据之间的均方跟误差
   * 
   * **/
  def computeRmse(model:MatrixFactorizationModel ,data:RDD[Rating], n:Long):Double = {
    
    val predictions:RDD[Rating] = model.predict((data.map(x => 
      (x.user,x.product))))
  
     val predictionAndratings = predictions.map { x => ((x.user,
         x.product),x.rating) }.join(data.map { 
           x => ((x.user,x.product),x.rating) }).values
     math.sqrt(predictionAndratings.map(x => (x._1-x._2)*(x._1-x._2))
         .reduce(_+_)/n)
  }

  
  def loadRating(path: String) : Seq[Rating] = {
    
    val lines = Source.fromFile(path).getLines()
    
    val ratings = lines.map { line => 
     val fileds = line.split("::")
     Rating(fileds(0).toInt,fileds(1).toInt,fileds(2).toDouble)      
    }.filter { _.rating > 0.0 }
    
    if(ratings.isEmpty){
     sys.error("No ratings provided.")
    }else{
      ratings.toSeq
    }
  }
}