---
title: "ML Ops"
center: true
theme: black
transition: slide
margin: 0.04
revealOptions:
   transition: 'slide'
css: 
- ./css/custom.css
---

<grid drop="5 20" drag="90 25" style="font-family:'Inter',sans-serif;background-color:#303030;border-radius:8px!important;padding:auto;align:center;">
# ML Ops Examples <!-- .element style="font-family:'Inter';color:#F8F8F8;" -->
    
#### <i class="fab fa-github"></i> [`saforem2/mlops`](https://www.github.com/saforem2/mlops)
</grid>

<grid drop="0 55" drag="100 30" style="line-height:0.6em;" align="top">
Sam Foreman <!-- .element style="font-family:'Inter';font-size:1.6em;font-weight:500;line-height:0.6;color:#E0E0E0!important;vertical-align:bottom!important;" -->

May, 2022 <!-- .element style="font-family:'Nanum Pen Script'; font-size:1.8em;color:#616161;vertical-align:top;font-weight:400;" -->
</grid>

<grid drag="100 10" drop="bottom" flow="row" align="left" style="font-size:1.5em;">

::: block <!-- .element style="margin-left:2%;margin-bottom:2%;" -->

[<i class="fab fa-github"></i>](https://github.com/saforem2)
[<i class="fas fa-home"></i>](https://samforeman.me)
[<i class="fab fa-twitter"></i>](https://twitter.com/saforem2)

:::

</grid>

<grid drag="30 30" drop="70 70" align="bottomright">
<img src="https://raw.githubusercontent.com/saforem2/anl-job-talk/main/docs/assets/anl.svg" width="100%" align="center" ></img>
</grid>

---

<grid drag="50 20" drop="top" align="center" style="margin-top:10%;">
 # 📊 ML Ops
</grid>

<grid drag="60 25" drop="center" align="center" class="note" bg="#66bb6a" style="text-align:left;">

**Goal**: Allow researchers to focus on their science / model development without all the boilerplate.

</grid>

---

<!-- .slide bg="white" -->

### MLOps

<grid drop="0 8" drag="100 92">

<iframe id="mlops" width="100%" height="100%" data-src="https://www.mlops.toys" data-preload data-background-interactive></iframe>

</grid>

---


<grid drag="40 40" drop="top">
[![](assets/logo_comet_dark.png)](https://comet.ml)
</grid>

<grid drag="100 50" drop="center">

```python
# For Comet to start tracking a training run,
# just add these two lines at the top of your training script:
import comet_ml

experiment = comet_ml.Experiment(
    api_key="API_KEY",
    project_name="PROJECT_NAME"
)
# Metrics from this training run will now be visible in the Comet UI
```
</grid>

---


# Why Comet?

::: block <!-- .element align="center" style="font-size:0.8em;" -->

::: block <!-- .element align="center" style="font-size:0.8em;" class="blockquote-wrapper" --> 

> While some ML platform vendors offer stand-alone experiment tracking or model production monitoring systems, Comet offers both.
> <br><span style="text-align:right!important;">&mdash; [comet.ml](https://comet.ml) </span>

:::

- Features:
    - Experiment tracking and management
    - Dataset versioning
    - Model registry
    - Model production monitoring
    - Code panels
    - Reports

:::

---

<!-- .slide bg="white" -->

# Comet

![](assets/comet-features.png) <!-- .element align="stretch" -->

---

# First Steps
1. Setup a Comet account
2. Get an API key
    - Make sure you're logged into comet.ml
    - Go to [settings](https://www.comet.ml/api/my/settings)
    - Developer Information --> **Generate API Key**
3. Install Comet
  ```bash
  python3 -m pip install comet_ml
  ```
4. Create an Experiment and log to Comet
  ```python
  from comet_ml import Experiment
  experiment = Experiment(
      api_key="API_KEY",
      project_name="PROJECT_NAME",
      workspace="WORKSPACE_NAME",
  )
  params = {'batch_size': 32, 'lr': 0.001}
  metrics = {'accuracy': 0.9, 'loss': 0.01}
  experiment.log_metrics(metrics, step=1)
  experiment.log_parameters(parameters)
  ```

---

#  Comet  `\(+\)` `TensorFlow`

- <i class="fab fa-github"></i> [`mlops/src/comet/tensorflow/main.py`](https:/www.github.com/saforem2/src/mlops/comet/tensorflow/main.py)
- [dashboard](https://www.comet.ml/saforem2/mlops/2cc1b07491554afcb42af1c7f040353e?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step)

---



<section data-background-iframe="https://www.comet.ml/saforem2/mlops/2cc1b07491554afcb42af1c7f040353e?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step" data-background-interactive></section>

note:
- Background iframe

---

# Comet `\(+\)` ⚡️`Pytorch Lightning`

---

# Comet `\(+\)`
- <i class="fab fa-github"></i>  [`mlops/src/comet/torch`](https://github.com/saforem2/mlops/src/comet/torch):
    - **DDP**: [`ddp.py`](https://github.com/saforem2/mlops/src/comet/torch/lightning.py)
    - ️️ [**Pytorch Lightning**](https://pytorch-lightning.rtfd.io/en/latest/) ⚡: [`lightning.py`](https://github.com/saforem2/mlops/src/comet/torch/lightning.py)
- <i class="fab fa-github"></i>  [`mlops/src/comet/tensorflow`](https://github.com/saforem2/mlops/src/comet/tensorflow):
    -  [`main.py`](https://github.com/saforem2/mlops/src/comet/tensorflow)

---
<!-- .slide bg="white" -->
<grid drop="0 0" drag="100 100">
<iframe width="100%" height="100%" data-src="https://www.comet.ml/saforem2/mlops/2cc1b07491554afcb42af1c7f040353e?experiment-tab=chart&showOutliers=true&smoothing=0&transformY=smoothing&xAxis=step" style="border:none;width:100%" data-preload data-background-interactive></iframe>
</grid>

---
<!-- .slide bg="white" -->
<grid drop="0 0" drag="100 100">
<iframe width="100%" height="100%" data-src="https://wandb.ai/l2hmc-qcd/l2hmc-qcd/reports/L2HMC-Report-04-04-2022---VmlldzoxNzgzODcx" style="border:none;width:100%" data-preload data-background-interactive></iframe>
</grid>

---

<style>
:root {
   --r-heading-font: 'Inter', sans-serif;
  font-size: 34px;
}
.horizontal_dotted_line{
  border-bottom: 2px dotted gray;
} 
.footer {
  font-size: 60%;
  vertical-align:bottom;
  color:#bdbdbd;
  font-weight:400;
  margin-left:-5px;
  margin-bottom:1%;
}
.note {
  padding:auto;
  margin: auto;
  text-align:center!important;
  border-radius: 8px!important;
  background-color: rgba(53, 53, 53, 0.5);
}
.reveal ul ul,
.reveal ul ol,
.reveal ol ol,
.reveal ol ul {
  margin-bottom: 10px;
}
.callout {
  background-color: #35353550
  color: #eeeeee;
}
.callout-content {
  overflow-x: auto;
  color: #eeeeee;
  background-color:#353535;
  padding: 5px 15px;
}
</style>