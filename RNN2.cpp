#include <cmath>
#include <iostream>
#include <vector>

class RNN {
 public:
  int input_size;        // 輸入層的大小
  int hidden_size;       // 隱藏層的大小
  int output_size;       // 輸出層的大小
  double learning_rate;  // 學習率

  // 權重和偏差
  std::vector<double> Wxh;  // 輸入到隱藏層的權重
  std::vector<double> Whh;  // 隱藏層到隱藏層的權重
  std::vector<double> Why;  // 隱藏層到輸出層的權重
  std::vector<double> bh;   // 隱藏層的偏差
  std::vector<double> by;   // 輸出層的偏差

  // 隱藏層狀態
  std::vector<double> h;  // 隱藏層狀態

  // 建構函數
  RNN(int input_size, int hidden_size, int output_size, double learning_rate)
      : input_size(input_size),
        hidden_size(hidden_size),
        output_size(output_size),
        learning_rate(learning_rate) {
    // 初始化權重和偏差
    Wxh.resize(input_size * hidden_size);   // 輸入到隱藏層的權重
    Whh.resize(hidden_size * hidden_size);  // 隱藏層到隱藏層的權重
    Why.resize(hidden_size * output_size);  // 隱藏層到輸出層的權重
    bh.resize(hidden_size);                 // 隱藏層的偏差
    by.resize(output_size);                 // 輸出層的偏差
    h.resize(hidden_size, 0.0);             // 隱藏層狀態初始化為 0

    // 隨機初始化權重和偏差
    for (auto& w : Wxh) w = ((double)rand() / RAND_MAX - 0.5) / hidden_size;
    for (auto& w : Whh) w = ((double)rand() / RAND_MAX - 0.5) / hidden_size;
    for (auto& w : Why) w = ((double)rand() / RAND_MAX - 0.5) / hidden_size;
    for (auto& b : bh) b = 0;
    for (auto& b : by) b = 0;
  }

  // 激活函數: sigmoid
  double sigmoid(double x) { return 1 / (1 + exp(-x)); }

  // 激活函數: sigmoid 的導數
  double sigmoid_derivative(double x) { return x * (1 - x); }

  // 激活函數: softmax
  std::vector<double> softmax(const std::vector<double>& logits) {
    std::vector<double> exp_logits(logits.size());
    double sum_exp = 0.0;

    // 計算每個元素的指數值及其總和
    for (size_t i = 0; i < logits.size(); ++i) {
      exp_logits[i] = exp(logits[i]);
      sum_exp += exp_logits[i];
    }

    // 計算 softmax
    std::vector<double> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
      probs[i] = exp_logits[i] / sum_exp;
    }

    return probs;
  }

  // 前向傳播
  std::vector<double> forward(const std::vector<double>& x) {
    std::vector<double> new_h(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
      new_h[i] = bh[i];
      for (int j = 0; j < input_size; ++j) {
        new_h[i] += x[j] * Wxh[i * input_size + j];
      }
      for (int j = 0; j < hidden_size; ++j) {
        new_h[i] += h[j] * Whh[i * hidden_size + j];
      }
      new_h[i] = sigmoid(new_h[i]);
    }
    h = new_h;

    std::vector<double> y(output_size);
    for (int i = 0; i < output_size; ++i) {
      y[i] = by[i];
      for (int j = 0; j < hidden_size; ++j) {
        y[i] += h[j] * Why[i * hidden_size + j];
      }
    }
    // 應用 softmax 函數
    return softmax(y);
  }

  // 訓練模型
  void train(const std::vector<std::vector<double>>& inputs,
             const std::vector<std::vector<double>>& targets) {
    for (int epoch = 0; epoch < 1000; ++epoch) {  // 訓練 1000 個迭代
      double loss = 0.0;
      for (size_t t = 0; t < inputs.size(); ++t) {
        std::vector<double> x = inputs[t];
        std::vector<double> target = targets[t];

        // 前向傳播
        std::vector<double> y = forward(x);

        // 計算損失（交叉熵損失） cross-entrophy
        loss = 0.0;  // 每個樣本的損失都需要重新計算
        for (size_t i = 0; i < y.size(); ++i) {
          if (target[i] > 0) {  // 確保只對正確標籤進行損失計算
            loss -= target[i] *
                    log(y[i] + 1e-10);  // 加入小的 epsilon 以避免 log(0)
          }
        }

        // 反向傳播
        std::vector<double> dy(output_size);
        for (size_t i = 0; i < output_size; ++i) {
          dy[i] = y[i] - target[i];
        }

        std::vector<double> dh(hidden_size, 0.0);
        for (size_t i = 0; i < hidden_size; ++i) {
          for (size_t j = 0; j < output_size; ++j) {
            dh[i] += dy[j] * Why[j * hidden_size + i];
            Why[j * hidden_size + i] -= learning_rate * dy[j] * h[i];
          }
          dh[i] *= sigmoid_derivative(h[i]);
        }

        std::vector<double> dWxh(input_size * hidden_size, 0.0);
        std::vector<double> dWhh(hidden_size * hidden_size, 0.0);
        for (size_t i = 0; i < hidden_size; ++i) {
          bh[i] -= learning_rate * dh[i];
          for (size_t j = 0; j < input_size; ++j) {
            dWxh[i * input_size + j] += dh[i] * x[j];
          }
          for (size_t j = 0; j < hidden_size; ++j) {
            dWhh[i * hidden_size + j] += dh[i] * h[j];
          }
        }

        for (size_t i = 0; i < input_size * hidden_size; ++i) {
          Wxh[i] -= learning_rate * dWxh[i];
        }
        for (size_t i = 0; i < hidden_size * hidden_size; ++i) {
          Whh[i] -= learning_rate * dWhh[i];
        }

        for (size_t i = 0; i < output_size; ++i) {
          by[i] -= learning_rate * dy[i];
        }
      }
      std::cout << "Epoch: " << epoch << ", Loss: " << loss << std::endl;
    }
  }
};

int main() {
  RNN rnn(2, 4, 3, 0.1);  // 輸入層為 2，隱藏層為 4，輸出層為 3

  // 假設這裡有一些輸入和目標資料
  std::vector<std::vector<double>> inputs = {
      {0, 0},   {0, 1}, {1, 0}, {1, 1},   {0.5, 1},   {1, 0.5},
      {0, 0.5}, {1, 0}, {0, 1}, {1, 0.5}, {0.5, 0.5}, {1, 1}};
  std::vector<std::vector<double>> targets = {
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
      {0, 0, 1}, {0, 0, 1}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};

  // 訓練 RNN 模型
  rnn.train(inputs, targets);

  return 0;
}
