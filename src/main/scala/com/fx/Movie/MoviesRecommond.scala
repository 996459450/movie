package com.fx.Movie

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.Contains
import org.apache.spark.sql.SQLContext
import com.sun.xml.internal.ws.wsdl.writer.document.Import
import java.util.Properties
import org.apache.spark.sql.SaveMode
//import com.mysql.jdbc.PreparedStatement
import java.sql.Connection
import java.sql.DriverManager
import java.sql.PreparedStatement


object MoviesRecommond {
 
  def main(args: Array[String]): Unit = {
    // 程序启动参数检测，当参数小于2时，程序不起动
    if(args.length < 2){
      System.err.print("Usage :")
      System.exit(1)
    }  
    //屏蔽日志为了保证能在client端看到结果
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
  Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  
  //配饰程序运行的上下文环境，并设置程序名字
  val conf = new SparkConf().setMaster(args(0)).setAppName("MoviesRecommond")
  
  val sc  = new SparkContext(conf)
  
  //评分数据集，已元组形式存在 (用户ID 电影ID 评分 时间戳)
  val ratingsList_tuple = sc.textFile(args(1)+"/ratings.dat").map { line => 
      val fileds = line.split("::")
      (fileds(0).toInt,fileds(1).toInt,fileds(2).toDouble,fileds(3).toLong%10)
    }
  
  //双元组 （时间戳，（用户ID，电影ID，评分））
  val ratingsTrain_KV = ratingsList_tuple.map(x => (x._4 ,Rating(x._1,x._2,x._3)))
  
  print(" get " +ratingsTrain_KV.count() +" ratings from " +ratingsTrain_KV
      .map(_._2.user).distinct().count()+" users on "+ ratingsTrain_KV
      .map(_._2.product).distinct().count()+" movies")
   
      //test.dat 文件  推荐用户文件 用户ID  电影ID 评分 时间戳
   val myRatedata_Rating = sc.textFile(args(2)).map { line =>  
        val fileds = line.split("::")
        Rating(fileds(0).toInt,fileds(1).toInt,fileds(2).toDouble)
    }
  
  val numParitions = 3
  val traningData_Rating = ratingsTrain_KV.filter(_._1 < 8)
      .values.union(myRatedata_Rating).repartition(numParitions).cache()
  val validateData_Rating = ratingsTrain_KV.filter(x => x._1 >= 6 && x._1 < 8)
  .values.repartition(numParitions).cache()
  
  val testData_Rating = ratingsTrain_KV.filter(_._1 >= 8).values.cache()
  println("training data is num : "+traningData_Rating.count() +" validate date is num "
      +validateData_Rating.count() +" test data is num : " +testData_Rating.count())
      
  val ranks = List(8,22)
  val lambds = List(0.1,10)
  val iters = List(5,7)
  var bestModel:MatrixFactorizationModel = null
  var bestValidateRnse = Double.MaxValue
  var bestRank = 0
  var bestLambda = -0.1
  var bestItere = -1
  
  for(rank <- ranks ; lam <- lambds ;iter <- iters ){
    val model = ALS.train(traningData_Rating, rank, iter , lam)
    val validateRnse = rnse(model, validateData_Rating, validateData_Rating.count())
    
    println(" validation= " + validateRnse 
        +" for the model training with rank = "+ rank +" lambda = " +lam 
        +" and numIter "+iter)
    
        if(validateRnse < bestValidateRnse){
          bestModel = model
          bestValidateRnse = validateRnse
          bestRank = rank
          bestLambda = lam
          bestItere = iter
        }
    }
  val testDataRnse = rnse(bestModel , testData_Rating, testData_Rating.count())

  println(" the best model model was trained with rank = "+bestRank + " and lambda = " 
      + bestLambda + " and numIter = " + bestItere + " and Rnse on the test data is " 
      + testDataRnse)
      
    val meanRating = traningData_Rating.union(validateData_Rating).map { _.rating }.mean()
    
     val baselineRnse = math.sqrt(testData_Rating.map { x => (meanRating
         - x.rating)* (meanRating - x.rating) }.mean())
         
   val improvent = (baselineRnse - testDataRnse) / baselineRnse * 100
   println("the best model improves the baseline by " + "%2.2f".format(improvent) + "%")
    
   val moviesList_Tuple = sc.textFile(args(1)+"/movies.dat").map { line =>
     val fileds = line.split("::") 
    (fileds(0).toInt,fileds(1),fileds(2)) 
    }
  val movies_Map = moviesList_Tuple.map(x => (x._1,x._2)).collect().toMap
 
  val moviesType_Map = moviesList_Tuple.map(x =>  (x._1, x._3)).collect().toMap

  var i = 1
  
  println("movies recommond for you:")
  val myRatedMoviesIds = myRatedata_Rating.map { _.product }.collect().toSet
  
    var conn: Connection = null
    var ps : PreparedStatement = null
    
   try{
      val recommondList = sc.parallelize(movies_Map.keys.filter {myRatedMoviesIds.contains(_)}.toSeq)
     bestModel.predict(recommondList.map {(0,_)}).collect().sortBy(-_.rating).take(10).foreach {
    
    r => println("%2d".format(i) + "----------> : \nmovie name --> "
        + movies_Map(r.product) + " \nmovie type --> "
        + moviesType_Map(r.product)) 
        i+=1    
    val sql = "insert into movies(id,moviename,movietype) values(?,?,?)"
    
     conn =  DriverManager.getConnection("jdbc:mysql://192.168.122.168:3306/movie", "root", "briup")
     ps = conn.prepareStatement(sql)
     ps.setString(1, (i-1).toString())
     ps.setString(2,movies_Map(r.product))
     ps.setString(3,moviesType_Map(r.product))
     ps.executeUpdate()
      }
    }catch{
    case e : Exception => println("mysql exception")
    }finally {
    	if(ps !=null){
    		ps.close()
    	}
    	if(conn != null){
    		ps.close()
    	}
    }
    
  /*
  println("you may be interested in these people : ")
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    
    //将电影，用户，评分数据转换成为DataFrame，进行SparkSQL操作
    val movies = moviesList_Tuple
      .map(m => Movies(m._1.toInt, m._2, m._3))
      .toDF()

    val ratings = ratingsList_tuple.map(r => 
      Ratings(r._1.toInt,r._2.toInt,r._3.toInt)).toDF()
      
     val users = sc.textFile(args(1) + "/users.dat").map { lines =>
      val fields = lines.split("::")
      Users(fields(0).toInt, fields(2).toInt, fields(3).toInt)
    }.toDF()

    ratings.filter('rating >= 5)//过滤出评分列表中评分为5的记录
      .join(movies, ratings("movieId") === movies("id"))//和电影DataFrame进行join操作
      .filter(movies("mType") === "Drama")//筛选出评分为5，且电影类型为Drama的记录（本来应该根据我的评分数据中电影的类型来进行筛选操作，由于数据格式的限制，这里草草的以一个Drama作为代表）
      .join(users, ratings("userId") === users("id"))//对用户DataFrame进行join
      .filter(users("age") === 18)//筛选出年龄=18（和我的信息一致）的记录
      .filter(users("occupation") === 15)//筛选出工作类型=18（和我的信息一致）的记录
      .select(users("id"))//只保存用户id，得到的结果为和我的个人信息差不多的，而且喜欢看的电影类型也和我差不多 的用户集合
      .take(10)
      .foreach(println)*/
  }
 
  def rnse(model:MatrixFactorizationModel ,predictionData :RDD[Rating],n:Long):Double = {
    
    val prediction = model.predict(predictionData.map { x => (x.user,x.product) })
    val preditionAndOldRatiing =prediction.map { x => ((x.user,x.product),x.rating)}
    .join(predictionData.map { x => ((x.user,x.product),x.rating) }).values
    math.sqrt(preditionAndOldRatiing.map(x => (x._1 - x._2)*(x._1-x._2)).reduce(_-_)/n)
  }
  
  case class Ratings(userId: Int, movieId: Int, rating: Int)

  case class Movies(id: Int, name: String, mType: String)

  case class Users(id: Int, age: Int, occupation: Int)

}