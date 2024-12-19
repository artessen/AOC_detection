# Defining the custom objective and metric functions ===========================
# Binary cross-entropy
loss_squared_log = function(y_pred, y_true) {
  y_true = as.numeric(getinfo(y_true, "label"))
  d = y_pred - y_true
  h = 5
  scale = 1 + (d / h)^2
  scale_sqrt = sqrt(scale)
  grad = d / scale_sqrt
  hess = 1 / scale / scale_sqrt
  return(list(grad = grad, hess = hess))
}

# precision - Custom metric function+
precision = function(y_pred, y_true, threshold=0.5) {
  y_true = as.numeric(getinfo(y_true, "label"))
  y_pred[y_pred>=threshold] = 1
  y_pred[y_pred< threshold] = 0
  pre = Metrics::precision(y_true, y_pred)
  if (is.nan(pre)) { pre = 0. }
  return(list(metric = "precision", value = pre))
}

# recall - Custom metric function+
recall = function(y_pred, y_true, threshold=0.5) {
  y_true = as.numeric(getinfo(y_true, "label"))
  y_pred[y_pred>=threshold] = 1
  y_pred[y_pred< threshold] = 0
  rec = Metrics::recall(y_true, y_pred)
  if (is.nan(rec)) { rec = 0. }
  return(list(metric = "recall", value = rec))
}

# f1_score - Custom metric function
f1_score = function(y_pred, y_true, threshold=0.5) {
  y_true = as.numeric(getinfo(y_true, "label"))
  y_pred[y_pred>=threshold] = 1
  y_pred[y_pred< threshold] = 0
  rec = Metrics::recall(y_true, y_pred)
  pre = Metrics::precision(y_true, y_pred)
  if (is.nan(rec)) { rec = 0. }
  if (is.nan(pre)) { pre = 0. }
  f1  = 2*((pre*rec)/(pre+rec))
  if (is.nan(f1)) { f1 = 0. }
  return(list(metric = "f1_score", value = f1))
}
# ==============================================================================