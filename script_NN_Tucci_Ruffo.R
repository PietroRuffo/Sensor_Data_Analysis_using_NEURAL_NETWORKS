
library(neuralnet)
library(caret)
library(NeuralNetTools)

my_data<-read.csv("data_NN.csv",sep="\t")

head(my_data)
dim(my_data)

my_data<-my_data[,c(5:ncol(my_data))]
head(my_data)
dim(my_data)
summary(my_data)

set.seed(123)
my_data<-my_data[sample(1:nrow(my_data),length(1:nrow(my_data))),1:ncol(my_data)]
head(my_data)

table(is.na.data.frame(my_data))
#non ci sono missing values

cor(my_data[,c(2:ncol(my_data))])
#no multicollinearità

attach(my_data)
str(my_data)

table(activity)
#il dataset è bilanciato

normalize<-function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

dati_scaled<-as.data.frame(lapply(my_data,normalize))
summary(dati_scaled)
str(dati_scaled)

set.seed(123)
index<-sample(1:nrow(my_data),round(0.70*nrow(my_data)))

train_data<-as.data.frame(my_data[index,])
train_data_scaled<-as.data.frame(dati_scaled[index,])

test_data<-as.data.frame(my_data[-index,])
test_data_scaled<-as.data.frame(dati_scaled[-index,])

n<-names(my_data)
n
f<-as.formula(paste("activity ~",paste(n[!n %in% "activity"],collapse=" + ")))
f

#act.fct: logistic, perchè per risolvere questo problema non vogliamo un gradiente troppo forte

#model
#hidden=5
model<-neuralnet(formula=f,data=train_data_scaled,hidden=5,act.fct="logistic",threshold=0.01,linear.output=FALSE,stepmax=1e7)

y_scaled<-compute(model,test_data_scaled)
previsioni<-y_scaled$net.result
head(previsioni)
x<-round(previsioni)
head(x)
#Validare la rete
conf<-confusionMatrix(as.factor(x),as.factor(test_data_scaled$activity))
conf #0.9842

#model2
#hidden=c(5,5)
model2<-neuralnet(formula=f,data=train_data_scaled,hidden=c(5,5),act.fct="logistic",threshold=0.01,linear.output=FALSE,stepmax=1e7)

y_scaled2<-compute(model2,test_data_scaled)
previsioni2<-y_scaled2$net.result
head(previsioni2)
x2<-round(previsioni2)
head(x2)
#Validare la rete --> non c'è bisogno di una rete con più di uno strato
conf2<-confusionMatrix(as.factor(x2),as.factor(test_data_scaled$activity))
conf2 #0.9825

#scegliamo model

#model3
#threshold=0.001
model3<-neuralnet(formula=f,data=train_data_scaled,hidden=5,act.fct="logistic",threshold=0.001,linear.output=FALSE,stepmax=1e7)

y_scaled3<-compute(model3,test_data_scaled)
previsioni3<-y_scaled3$net.result
head(previsioni3)
x3<-round(previsioni3)
head(x3)
#Validare la rete
conf3<-confusionMatrix(as.factor(x3),as.factor(test_data_scaled$activity))
conf3 #0.9883

#scegliamo model3

plotnet(model3,cex=0.8)
#numero di parametri: 6*5 +5*1 + 5 + 1 = 41


library(ROCR)
f_prediction<-prediction(x3,test_data_scaled$activity)
f_performance<-performance(f_prediction,"tpr","fpr")
plot(f_performance,colorize=TRUE,main="ROC for Neural Network")
performance(f_prediction,"auc")@y.values #area sotto curva


#metodo antagonista: modello di Regressione Logistica
mod<-glm(formula=f,data=train_data_scaled,family=binomial(link=logit))
summary(mod)

previsioni<-predict(mod,test_data_scaled,type="response")
#Validare modello logistico
conf_reg<-confusionMatrix(as.factor(round(previsioni)),as.factor(test_data_scaled$activity))
conf_reg

f_prediction_reg<-prediction(round(previsioni),test_data_scaled$activity)
f_performance_reg<-performance(f_prediction_reg,"tpr","fpr")
plot(f_performance_reg,colorize=TRUE,main="ROC for Logistic Regression")
performance(f_prediction_reg,"auc")@y.values #area sotto curva

#confronto tra rete e mod log

conf3
conf_reg

par(mar=c(5.1,4.1,6,2.1), xpd=TRUE)
plot(f_performance,col="blue")
plot(f_performance_reg,col="red",add=T)
legend("topright",legend=c("Neural Network","Logistic Regression"),lty=1,col=c("blue","red"),bty="n",inset=c(0,-0.5),cex=0.75)
performance(f_prediction,"auc")@y.values
performance(f_prediction_reg,"auc")@y.values

