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
    // 2 X 4 X 4 X 1
    Wxh.resize(input_size * hidden_size);   // 初始化 Wx 到隱藏層
    Whh.resize(hidden_size * hidden_size);  // 初始化 Wh 到隱藏層
    Why.resize(hidden_size * output_size);  // 初始化 Wh 到輸出層
    bh.resize(hidden_size);                 // 初始化隱藏層偏差
    by.resize(output_size);                 // 初始化輸出層偏差
    h.resize(hidden_size, 0.0);             // 初始化隱藏層狀態為 0

    // 初始化權重和偏差
    for (auto& w : Wxh) w = ((double)rand() / RAND_MAX - 0.5) / hidden_size;
    for (auto& w : Whh) w = ((double)rand() / RAND_MAX - 0.5) / hidden_size;
    for (auto& w : Why) w = ((double)rand() / RAND_MAX - 0.5) / hidden_size;
    for (auto& b : bh) b = 0;
    for (auto& b : by) b = 0;
  }

  // 激活函數
  double sigmoid(double x) { return 1 / (1 + exp(-x)); }

  // 激活函數的導數
  double sigmoid_derivative(double x) { return x * (1 - x); }

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
      y[i] = sigmoid(y[i]);
    }
    return y;
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

        // 計算損失
        // MSE版本
        for (size_t i = 0; i < y.size(); ++i) {
          loss += 0.5 * (target[i] - y[i]) * (target[i] - y[i]);
        }

        // 計算損失
        // cross-entrophy版本
        // for (size_t i = 0; i < y.size(); ++i) {
        //   if (target[i] > 0) {  // 確保只對正確標籤進行損失計算
        //     loss -= target[i] * log(y[i]);
        //   }
        // }

        // 反向傳播
        std::vector<double> dy(output_size);
        for (size_t i = 0; i < output_size; ++i) {
          dy[i] = (y[i] - target[i]) * sigmoid_derivative(y[i]);
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
  RNN rnn(2, 4, 1, 0.1);

  // 假設這裡有一些輸入和目標資料
  std::vector<std::vector<double>> inputs = {
      {0, 0},     {0, 1},    {1, 0},     {1, 1},    {1, 0.5}, {0.5, 1},
      {0.5, 0.5}, {0, 0.5},  {1, 0.75},  {0.75, 1}, {0.5, 1}, {0.25, 1},
      {1, 0.5},   {1, 0.75}, {0.5, 0.5}, {1, 0},    {0, 1},   {1, 0}};
  std::vector<std::vector<double>> targets = {
      {0},    {1}, {1},    {0}, {1}, {1},   {0.5}, {0.5}, {1},
      {0.75}, {1}, {0.75}, {1}, {1}, {0.5}, {1},   {1},   {1}};

  // 訓練 RNN 模型
  rnn.train(inputs, targets);

  return 0;
}

/*
RNN 模型結構

    輸入層（Input Layer）：輸入大小為2。
    隱藏層（Hidden Layer）：隱藏層大小為4。
    輸出層（Output Layer）：輸出大小為1。
    學習率（Learning Rate）：設定為0.1。

權重和偏差

    Wxh：輸入到隱藏層的權重矩陣，大小為 (input_size * hidden_size)。
    Whh：隱藏層到隱藏層的權重矩陣，大小為 (hidden_size * hidden_size)。
    Why：隱藏層到輸出層的權重矩陣，大小為 (hidden_size * output_size)。
    bh：隱藏層的偏差向量，大小為 hidden_size。
    by：輸出層的偏差向量，大小為 output_size。

前向傳播（Forward Propagation）

    計算新的隱藏層狀態 new_h：
        new_h 初始值為隱藏層的偏差 bh。
        加上輸入 x 經過 Wxh 的影響。
        加上先前隱藏層狀態 h 經過 Whh 的影響。
        經過 sigmoid 函數進行非線性變換。

    計算輸出 y：
        y 初始值為輸出層的偏差 by。
        加上隱藏層狀態 h 經過 Why 的影響。
        經過 sigmoid 函數進行非線性變換。

反向傳播（Backpropagation）

    計算輸出層的誤差 dy：
        dy 為預測輸出 y 與目標值 target 之間的差異，乘以 sigmoid 導數。

    計算隱藏層的誤差 dh：
        dh 為 dy 經過 Why 的反向傳播影響，乘以隱藏層狀態 h 的 sigmoid 導數。

    更新權重 Why、Wxh 和 Whh 以及偏差 by 和 bh：
        根據誤差值，使用學習率進行梯度下降。

訓練過程

    訓練迭代 1000
次，每次迭代會對所有輸入資料進行一次前向傳播和反向傳播，並更新權重和偏差。
    計算並輸出每個 epoch 的損失。

主函數（main 函數）

    定義了 inputs 和 targets 資料集。
    建立 RNN 模型並進行訓練。


模型圖示
以下是模型結構的示意圖：
輸入層 (Input Layer) - 2個節點
     |
     V
隱藏層 (Hidden Layer) - 4個節點
     |
     V
輸出層 (Output Layer) - 1個節點

*/
