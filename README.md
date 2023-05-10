# ultrasoud-trace-tipneedle
 
## 标定针尖
求解模板坐标初始化，并利用梯度下降法进行模板参数的优化：
``Others/build_template.py``

求优化针尖坐标``Tip_calibration/CalNeedleTip.py``，运行``Tip_calibration/main.py``输出针尖在模板坐标系和世界坐标系下的坐标。

> 预计效果：模板坐标求取之后，转换成yaml文件存储起来，符合4小球的模板都能直接只用此坐标轴。

使用数据：``YamlFiles/experience_data2.yaml``
## 标定超声波

pass
