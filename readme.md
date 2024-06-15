### target

seq2seq，non-causal(至少处理某时效数据时可以参考当天预测后续时效的结果)

<img width="843" alt="截屏2024-06-16 02 20 08" src="https://github.com/hobeter/tz_/assets/55823642/28482973-e5b5-446d-b9f9-dab13f534712">
<img width="677" alt="截屏2024-06-16 02 19 41" src="https://github.com/hobeter/tz_/assets/55823642/1628c6f8-e98f-4847-8151-128c5c907c1f">


### data_preprocess

调用data_preprocess中的`prepate_data`返回对应的对齐、(按预测日隔离后)切分过的训练数据。调用示例见`demo_train_torch.py`。

对一种模型(树、高斯过程、网络等)，可通过调整训练数据实现多角度模型集成，例如：

- 通过调整`indicator_list`，实现单站点、不同站点联合的多模型集成；
- 通过调整`feature_list`，实现单特征、不同特征联合的多模型集成；
- 通过调整`stride`，实现多时长的多个模型集成；
- 通过调整`overlap`，实现推理时单一模型多次预测的集成；
...

```python
def prepare_data(
    indicator_list: List[str] = [], # stn_id
    feature_list: List[str] = [], # feature
    eval_ratio: float = 0.1,
    stride: int = 4,
    overlap: int = 0,
    null_padding: float = 0.0
)-> dict:
"""
return {
    "indicator_list": indicator_list, # 观测点表
    "feature_list": feature_list, # 特征表
    "eval_ratio": eval_ratio,
    "stride": stride, # 窗口步长
    "overlap": overlap, # 窗口重叠
    "null_padding": null_padding,
    "train_data": [
        {   
            "date": date,
            "start_idx": start_idx,
            "indicator_list": indicator_list,
            "dates": dates, # 预测的时间列表
            "feature_list": feature_list,
            "stride": stride,
            "overlap": overlap,
            "metadata": metadata,


            "gathered_input_ave": gathered_input_ave, # 预测特征均值列表 (feature)

            # 以下都是(num, length, pos, feature)的列表
            "gathered_input": gathered_input, # 预测值(可作为输入)
            "gathered_target_abs": gathered_target_abs, # 观测值(可作为label)
            "gathered_target_delta_pred": gathered_target_delta_pred, # 修正数值(相对于预测值)(可作为label)
            "gathered_target_delta_ave": gathered_target_delta_ave # 修正数值(相对于均值)(可作为label)
        },
        ...
    ]
    "eval_data": 同上
}
"""
```
