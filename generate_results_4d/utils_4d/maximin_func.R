
suppressMessages(library(maximin))

#' Maximin distance design with previously existing data
maxmin_dist <- function(n, p, T, Xorig=NULL, verb=FALSE, boundary=FALSE){
  
	Xsparse <- maximin(n, p, T, Xorig, Xinit=NULL, verb, plot=FALSE, boundary)

	return(Xsparse)
  
}