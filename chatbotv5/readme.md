## 训练
python demo.py train

## 预测
python demo.py predict

#### 效果如下：
<pre>
[root@localhost $] python demo.py predict
Building prefix dict from the default dictionary ...
Loading model from cache
/var/folders/tq/c0kp5y857x138x5vf8bzxfc80000gp/T/jieba.cache
Loading model cost 0.969 seconds.
Prefix dict has been built succesfully.
> 你好
你 也好 ~
> 呵呵
傻 逼 呵呵
> 哈哈
笑 屁
> 你是谁
我 是 小猴子
> 早
WARN：词汇不在服务区
> 早上好
哈哈
> 屁
你 屁 会
> 滚蛋
WARN：词汇不在服务区
> 傻逼
他妈 逼 的
</pre>

## 代码讲解
欢迎关注[www.shareditor.com](http://www.shareditor.com/)最新文章
