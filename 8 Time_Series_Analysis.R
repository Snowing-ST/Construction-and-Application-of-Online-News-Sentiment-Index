
#总指数与传统消费者信心指数的多元时间序列分析---------

library(forecast)
library(ggplot2)
library(TSA)

setwd("E:/graduate/Paper/对比指标")
data = read.csv("CCCI.csv")
# data = read.csv("CCCI_CNN.csv")
data = na.omit(data)
head(data)

#线性模型

data1 = ts(data[,-c(1,2)],frequency = 4,start = c(2009,1))
fit = tslm(CCCI ~Index,data1)
summary(fit)
#拟合图
autoplot(data1[,"CCCI"], series="Data") +
  autolayer(fitted(fit), series="Fitted") +
  xlab("Year") + ylab("") +
  ggtitle("") +
  guides(colour=guide_legend(title=" "))

#拟合值相关系数 0.837513没有好到哪里去
cor(fitted(fit),data1[,"CCCI"])
#模型稳定性检验：残差是否平稳？？
print(kpss.test(fit$residuals,null="Level"))#不能推翻平稳的原假设
print(kpss.test(fit$residuals,null="Trend"))#趋势非平稳
adf.test(fit$residuals) #不能推翻非平稳的原假设
# 残差存在自相关，正态性？？
checkresiduals(fit) #LM检验原假设：白噪声
acf(fit$residuals,lag.max=12)
dwtest(CCCI ~Index,data=data1)#dw检验只能检验一阶自相关，原假设不存在一阶序列相关
qqnorm(fit$residuals)
qqline(fit$residuals)
#LM检验原假设为：直到p阶滞后不存在序列相关，p为预先定义好的整数；备选假设是：存在p阶自相关。
CV(fit)
accuracy(fit)

cbind(fitted(fit),data1)
#加入季节项、趋势项，都不显著,调整的R方反而小了
fit1 = tslm(CCCI ~trend+season+Index,data1)
summary(fit1)
print(kpss.test(fit1$residuals,null="Level"))#不能推翻平稳的原假设
print(kpss.test(fit1$residuals,null="Trend"))#趋势非平稳
# 残差不存在自相关
checkresiduals(fit1)
CV(fit1) #AIC反而高了，不好。。
accuracy(fit1)

#既然残差存在一阶自相关，那就放入CCCI的一阶滞后项
library(dyn)
fit2 = dyn$lm(CCCI ~Index+lag(CCCI,-1),data1)
summary(fit2)
checkresiduals(fit2)
accuracy(fit2)
cor(fit2$fitted.values,fit2$model$CCCI)

# ARIMAX-------------------------
#假定误差项能拟合arima模型
#误差项自动拟合arima
(fit.arima <- auto.arima(data1[,"CCCI"],
                         xreg=data1[,"Index"]))
#回归误差和回归误差拟合arima后的误差
cbind("Regression Errors" = residuals(fit.arima, type="regression"),
      "ARIMA errors" = residuals(fit.arima, type="innovation")) %>%
  autoplot(facets=TRUE)

# Ljung-Box test 原假设：白噪声
checkresiduals(fit.arima)
tsdiag(fit.arima)
cor(fit.arima$fitted,data1[,"CCCI"]) 
accuracy(fit.arima)
#拟合图
autoplot(data1[,"CCCI"], series="Data") +
  autolayer(fit.arima$fitted, series="Fitted") +
  xlab("Year") + ylab("") +
  ggtitle("") +
  guides(colour=guide_legend(title=" "))

cbind(fit.arima$fitted,data1)

#VAR--------------------------------------------------
#单位根检验
# 原序列kpss检验在95%水平显著，非平稳
# 一阶差分序列在95%水平不显著，平稳
# 一阶单整
library(urca)
library(tseries)
library(vars)
# library(devtools)
# install_github("cran/MSBVAR")
library(MSBVAR)


setwd("E:/graduate/Paper/对比指标")
data = read.csv("CCCI.csv")
# data = read.csv("CCCI_CNN.csv")
data = na.omit(data)
head(data)

#stl

datastl = stl(ts(data[,4],frequency = 4,start = c(2009,01)),"per")
plot(datastl,main="STL decomposition")

for(i in 2:4)
{
  print(kpss.test(data[,i],null="Level"))
  print(kpss.test(data[,i],null="Trend"))
  print(kpss.test(diff(data[,i]),null="Level"))
  print(kpss.test(diff(data[,i]),null="Trend"))
  
}


# data2 =data[,-c(1,3)]
data2 =ts(data[,-c(1,2)],frequency = 4,start = c(2009,01))
#协整检验
# EG两步法
#所有的回归结果都显著
z = list()->z1
for(i in 1:2)z[[i]] = lm(data2[,i]~.,data.frame(data2[,-i]))
for(i in 1:2)print(summary(z[[i]]))
#对回归残差的DF检验都显著，说明残差都为I(0)，可能存在协整关系
for(i in 1:2)
{
  z1[[i]] = ur.df(z[[i]]$residuals)
  if(i==1)print(z1[[i]]@cval);print(z1[[i]]@teststat)
}

VARselect(data2, lag.max = 4, type = "both",season = 4) 
#模型平稳性检验
p1ct <- VAR(data2, p = 2, type =  "both",season=4)    
p1ct 
plot(p1ct)

plot(stability(p1ct, type = c("OLS-CUSUM"), h = 0.15, dynamic = FALSE, rescale = TRUE))


# computes the multivariate Portmanteau- and Breusch-Godfrey test for serially correlated errors              
ser11 <- serial.test(p1ct, lags.pt = 16, type = "PT.asymptotic")
ser11$serial     
# The null hypothesis in portmanteau test is that the residuals are serially uncorrelated.

# The null hypothesis in normality test is that the residuals are subject to normal-distributed.
norm1 <- normality.test(p1ct)              
norm1$jb.mul
plot(norm1)

#granger test
round(granger.test(data2,p=1),4)

#拟合
fitv = ts(fitted(p1ct)[,1],frequency = 4,end = c(2018,02))
real = ts(data2[-c(1:(dim(data2)[1]-length(fitv))),"CCCI"],frequency = 4,end = c(2018,02))
cor(fitv,real ) 
#RMSE
sqrt(mean((fitv-real)^2))
#MAE
mean(abs(fitv-real))
autoplot(real, series="Data") +
  autolayer(fitv, series="Fitted") +
  xlab("Year") + ylab("") +
  ggtitle("") +
  guides(colour=guide_legend(title=" "))

cbind(fitv,real)
#进阶VARX--------------------------------------------------
library(dse)
#在R的dse包中
#The ARMA model representation is general, 
#so that VAR, VARX,ARIMA, ARMAX, ARIMAX can all be considered to be special cases
setwd("E:/graduate/Paper/对比指标")
data = read.csv("CCCI.csv")
data = na.omit(data)
head(data)

input<-ts(data[,4],frequency = 4,start = c(2009,1))
output<-ts(data[,3],frequency = 4,start = c(2009,1))
par(mfrow=c(2,1))
plot.ts(input)
plot.ts(output)
par(mfrow=c(1,1))
# 做两个变量间的协方差函数需要使用ccf函数
ccf(output,input,lag.max =NULL,type = c("correlation"),plot = TRUE)
# 可以看到两变量不存在逻辑上的因果关系，变量延迟的阶数为0
#将数据转化为需要的格式，这里要使用TSdata函数，把input与output数据存储在一起
armaxdata<-TSdata(input=input,output=output)
seriesNamesInput(armaxdata) = "Index"
seriesNamesOutput(armaxdata) = "CCCI"
#求解函数estVARXls，采用的求解方法为ls，该函数的好处是可以自动定阶
model <- estVARXls(armaxdata,max.lag = 2)
summary(model) 
print(model)
#由于程序包vars也有同名函数stability，要先解除
detach(package:vars)
stability(model)
tfplot(x=model)

cor(model$estimates$pred[-c(1,2)],model$data$output[-c(1,2)])
#RMSE
sqrt(mean((model$estimates$pred[-c(1,2)]-model$data$output[-c(1,2)])^2))
#MAE
mean(abs(model$estimates$pred[-c(1,2)]-model$data$output[-c(1,2)]))


cbind(model$estimates$pred,data1)

#手动定阶？？？不知怎么设计参数
MA <- array(c(1,-1.5,0.5),c(3,1,1))
C <- array(c(0,0,0,-0.5,-0.5,-0.5),c(5,1,1))
AR <- array(c(1,-0.5),c(2,1,1))
TR <-array(c(50),c(1,1,1))
ARMAX <- ARMA(A=AR, B=MA, C=C ,TREND<-TR)
model <- estMaxLik(armaxdata, ARMAX, algorithm="optim")
model
# A(L)y(t) = B(L)w(t) + C(L)u(t) + TREND(t)
L是延迟算子
# where
# A (axpxp) is the auto-regressive polynomial array.
# B (bxpxp) is the moving-average polynomial array.
# C (cxpxm) is the input polynomial array. C should be NULL if there is no input
# y is the p dimensional output data.
# u is the m dimensional control (input) data.
# TREND is a matrix the same dimension as y, a p-vector (which gets replicated for each time
# period), or NULL.

#状态空间VARX模型 更上方拟合效果完全一致

model.ss = estSSfromVARX(armaxdata,max.lag=2)
print(model.ss)
detach(package:vars)
stability(model.ss)
tfplot(x=model.ss)
#RMSE
sqrt(mean((model.ss$estimates$pred[-c(1,2)]-model$data$output[-c(1,2)])^2))
#MAE
mean(abs(model.ss$estimates$pred[-c(1,2)]-model$data$output[-c(1,2)]))
cor(model.ss$estimates$pred[-c(1,2)],model$data$output[-c(1,2)])


#------------
#------------
# 分指数与传统消费者信心指数的多元时间序列分析--------
# 平稳性检验---------------------
library(urca)
library(tseries)
# library(devtools)
# install_github("cran/MSBVAR")
library(MSBVAR)

setwd("E:/graduate/Paper")
data = read.csv("TS-REG.csv")
head(data)


for(i in 4:11)
{
  print(kpss.test(data[,i],null="Level"))
  print(kpss.test(data[,i],null="Trend"))
  print(kpss.test(diff(data[,i]),null="Level"))
  print(kpss.test(diff(data[,i]),null="Trend"))
  
}

#house不能通过协整检验，剔除或者用逆差分的方法将其变成单整序列
HOUSE = rep(0,dim(data)[1])

for(i in 4:length(HOUSE))
{
  HOUSE[i] = data[i,9]+data[i-1,9]+data[i-2,9]+data[i-3,9]
}
data[,9] = HOUSE
data[1:4,9] = rep(NA,4)
names(data)[9] = "HOUSE"
# 协整检验-------------------
# data2 =data[,-c(1,3)]
data2 =ts(data[,-c(1,2,3,11)],frequency = 4,start = c(2009,01))
head(data2)
#josenhan
# 拒绝r<=4的原假设，但不能拒绝r<=5的原假设
at = ca.jo(data2,type="trace",ecdet="const",season = 12)
at@cval;at@teststat
# 拒绝r<=4的原假设，但不能拒绝r<=5的原假设，至少存在5组长期协整关系。
ae = ca.jo(data2,type="eigen",ecdet="const",season = 12)
ae@cval;ae@teststat


# granger test-------------------
round(granger.test(data2,p=1),4)
round(granger.test(data2,p=2),4)


# 回归分析--------------------------------------------------
library(forecast)
library(ggplot2)
library(TSA)
library(corrplot)

setwd("E:/graduate/Paper")
data = read.csv("TS-REG.csv")
head(data)
#假定误差项是白噪声

data1 = data[,-c(1,2,3,9,11)]#去掉平稳的购房
head(data1)
data1 = ts(data1,frequency = 4,start = c(2009,1))
#变量之间无明显相关性
r = cor(na.omit(data1))
corrplot.mixed(r,lower="number",upper="color",addrect = 3,tl.col = "black",tl.cex = 0.7,rect.col = "black")
#线性模型
fit = tslm(CCCI ~ employment+investment+prices+livelihood+economics+house,data1)
summary(fit)
#拟合图
autoplot(data1[,"CCCI"], series="Data") +
  autolayer(fitted(fit), series="Fitted") +
  xlab("Year") + ylab("") +
  ggtitle("") +
  guides(colour=guide_legend(title=" "))

#比直接平均的总指数，相关度高
cor(fitted(fit),data1[,"CCCI"])
#模型稳定性检验：残差是否平稳？？
print(kpss.test(fit$residuals,null="Level"))
print(kpss.test(fit$residuals,null="Trend"))
# 残差存在自相关，正态性？？
checkresiduals(fit)
qqnorm(fit$residuals)
qqline(fit$residuals)
#LM检验原假设为：直到p阶滞后不存在序列相关，p为预先定义好的整数；备选假设是：存在p阶自相关。
CV(fit)
accuracy(fit)

#加入季节项、趋势项，adR方提升0.1
fit1 = tslm(CCCI ~trend+season+employment+investment+prices+livelihood+economics,data1)
summary(fit1)
# 残差存在自相关
checkresiduals(fit1)
CV(fit1) #反而低了。。


# 逐步回归选择变量，adR方再提升0.1
step(fit1)
fit.step  = step(fit1)
summary(fit.step)
CV(fit.step)
checkresiduals(fit.step)
cor(fitted(fit.step),data1[,"CCCI"]) #比纯线性回归还降低了0.03。。
#拟合图
autoplot(data1[,"CCCI"], series="Data") +
  autolayer(fitted(fit.step), series="Fitted") +
  xlab("Year") + ylab("") +
  ggtitle("") +
  guides(colour=guide_legend(title=" "))

accuracy(fit.step)


#考虑CCCI的一阶自相关
library(dyn)
fit.lag = dyn$lm(CCCI ~employment+investment+prices+livelihood+economics+lag(CCCI,-1),data=data1)
summary(fit.lag)

checkresiduals(fit.lag)
accuracy(fit.lag)
cor(fit.lag$fitted.values,fit.lag$model$CCCI)

#逐步回归
#转化为tslm模型，因为dyn不能用step
lagdata = fit.lag$model
names(lagdata)[7:length(names(lagdata))] = c("CCCI.lag1")
lagdata = ts(lagdata,frequency = 4,start = c(2009,03))
fit.lag1 = tslm(CCCI ~employment+investment+prices+livelihood+economics+CCCI.lag1,lagdata)
summary(fit.lag1)


fit.step.lag  = step(fit.lag1)
summary(fit.step.lag)
accuracy(fit.step.lag)
checkresiduals(fit.step.lag)
cor(fitted(fit.step.lag),fit.step.lag$model$CCCI) 


#ARIMAX-------------------
#假定误差项能拟合arima模型
#误差项自动拟合arima,加入趋势项t
(fit.arima <- auto.arima(data1[,"CCCI"],ic = "aicc",stepwise = TRUE,
                   xreg=cbind(data1[,-dim(data1)[2]],1:dim(data1)[1])))
#回归误差和回归误差拟合arima后的误差
cbind("Regression Errors" = residuals(fit.arima, type="regression"),
      "ARIMA errors" = residuals(fit.arima, type="innovation")) %>%
  autoplot(facets=TRUE)

# Ljung-Box test 原假设：白噪声
checkresiduals(fit.arima)
tsdiag(fit.arima)
cor(fit.arima$fitted,data1[,"CCCI"])
accuracy(fit.arima)
#拟合图
autoplot(data1[,"CCCI"], series="Data") +
  autolayer(fit.arima$fitted, series="Fitted") +
  xlab("Year") + ylab("") +
  ggtitle("") +
  guides(colour=guide_legend(title=" "))


#手动拟合
res = residuals(fit.arima, type="regression")
## 平稳行检验 差分平稳
print(kpss.test(res,null="Level"))
print(kpss.test(res,null="Trend"))
print(kpss.test(diff(res),null="Level"))
print(kpss.test(diff(res),null="Trend"))
#模型识别 arima(0,1,1)
par(mfrow = c(2,2))
acf(res);pacf(res);acf(diff(res,1));pacf(diff(res,1))
par(mfrow=c(1,1))

res.fit = Arima(res,order=c(1,0,0),method="ML");res.fit
#对比自动拟合的好一点点
auto.arima(res)

fit1=Arima(data1[,"CCCI"],order=c(0,1,1),seasonal=list(order=c(1,0,0),period=4),xreg=data1[,-7],method="ML");fit1
cbind("Regression Errors" = residuals(fit1, type="regression"),
      "ARIMA errors" = residuals(fit1, type="innovation")) %>%
  autoplot(facets=TRUE)

checkresiduals(fit1)
tsdiag(fit1)
cor(fit1$fitted,data1[,"CCCI"]) #比自动拟合提高一丢丢0.9364
accuracy(fit1)


#如果仅用step中挑选的变量
(fit2 <- auto.arima(data1[,"CCCI"],
                   xreg=data1[,names(fit.step$coefficients)[-1]]))
#回归误差和回归误差拟合arima后的误差
cbind("Regression Errors" = residuals(fit2, type="regression"),
      "ARIMA errors" = residuals(fit2, type="innovation")) %>%
  autoplot(facets=TRUE)

# Ljung-Box test 原假设：白噪声
checkresiduals(fit2)
tsdiag(fit2)
cor(fit2$fitted,data1[,"CCCI"]) #低了。。
accuracy(fit2)

# VAR------------------------------------------------------
library(vars)

setwd("E:/graduate/Paper")
data = read.csv("TS-REG.csv")
head(data)

data1 =ts(data[,-c(1,2,3,9,11)],frequency = 4,start = c(2009,01))#去掉house
head(data1)

VARselect(data1, lag.max = 2, type = "both",season = 4) #好多NA？？
#模型平稳性检验
p1ct <- VAR(data1, p = 2, type =  "both",season=4)    
# p1ct 
# plot(p1ct)

plot(stability(p1ct, type = c("OLS-CUSUM"), h = 0.15, dynamic = FALSE, rescale = TRUE))


# computes the multivariate Portmanteau- and Breusch-Godfrey test for serially correlated errors              
ser11 <- serial.test(p1ct, lags.pt = 16, type = "PT.asymptotic")
ser11$serial     
# The null hypothesis in portmanteau test is that the residuals are serially uncorrelated.

# The null hypothesis in normality test is that the residuals are subject to normal-distributed.
norm1 <- normality.test(p1ct)              
norm1$jb.mul
plot(norm1)


#拟合
fitv = fitted(p1ct)[,dim(fitted(p1ct))[2]]
#滞后一阶，少了第一个数
real = p1ct$y[(dim(p1ct$y)[1]-length(fitv)+1):dim(p1ct$y)[1],"CCCI"]
cor(fitv,real) 
#RMSE
sqrt(mean((fitv-real)^2))
#MAE
mean(abs(fitv-real))
autoplot(ts(real,frequency = 4,end = c(2018,02)), series="Data") +
  autolayer(ts(fitv,frequency = 4,end = c(2018,02)), series="Fitted") +
  xlab("Year") + ylab("") +
  ggtitle("") +
  guides(colour=guide_legend(title=" "))

cbind(fitv,1:length(fitv))


#VARX-------------------
library(dse)

setwd("E:/graduate/Paper")
data = read.csv("TS-REG.csv")
head(data)


input<-ts(data[,4:8],frequency = 4,start = c(2009,1)) #去掉house
output<-ts(data[,10],frequency = 4,start = c(2009,1))
# par(mfrow=c(2,1))
# plot.ts(input)
# plot.ts(output)
# par(mfrow=c(1,1))
# # 做两个变量间的协方差函数需要使用ccf函数
# ccf(output,input,lag.max =NULL,type = c("correlation"),plot = TRUE)
# 可以看到两变量不存在逻辑上的因果关系，变量延迟的阶数为0
#将数据转化为需要的格式，这里要使用TSdata函数，把input与output数据存储在一起
armaxdata<-TSdata(input=input,output=output)
# seriesNamesInput(armaxdata) = names(data[,4:7])
# seriesNamesOutput(armaxdata) = "CCCI"
#求解函数estVARXls，采用的求解方法为ls，该函数的好处是可以自动定阶
model <- estVARXls(armaxdata,max.lag = 2)
# summary(model) 
print(model)
#由于程序包vars也有同名函数stability，要先解除
# detach(package:vars)
# stability(model)
# tfplot(x=model)

#RMSE
sqrt(mean((model$estimates$pred[-c(1,2)]-model$data$output[-c(1,2)])^2))
#MAE
mean(abs(model$estimates$pred[-c(1,2)]-model$data$output[-c(1,2)]))
cor(model$estimates$pred[-c(1,2)],model$data$output[-c(1,2)])



