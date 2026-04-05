@main def dumpGraphs(methodFilter: String = ".*"): Unit = {
  cpg.method.name(methodFilter).l.foreach { method =>
    println(s"METHOD\t${method.name}")
    method.dotAst.l.foreach(dot => println(s"AST\t${dot}"))
    method.dotCfg.l.foreach(dot => println(s"CFG\t${dot}"))
    method.dotPdg.l.foreach(dot => println(s"PDG\t${dot}"))
  }
}

