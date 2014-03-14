#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <random>

class ICA {
  public:
    static constexpr int kNumData = 500;
    static constexpr int kRepeatNum = 200;
    static constexpr int kNumMatrix = 2;
    static constexpr int kNumK = 100;
    static constexpr double kMu = 1.0;
    static constexpr double kConvergenceError = 1.0 * pow(10, -8);
    static constexpr int kMaxNumIterations = 100;
    void Calculation();

  private:
    void GenerateRandNum(std::vector<double> &);
    void MakeSource(std::vector<double> &, std::vector<double> &,
        std::vector<double> &, std::vector<double> &);
    void EigenJacobiMethod(std::vector<std::vector<double> > &,
        std::vector<std::vector<double> > &);
    double H4(double &);
    double H3(double &);
    void GradientAlgorithm(std::vector<double> &, std::vector<double> &);
};

void ICA::GenerateRandNum(std::vector<double> &rand_num) {
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution<double> u_dist(-1.0, 1.0);
  std::vector<double> uniform_rand(kNumData, 0.0);

  for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
    uniform_rand[n_cnt] = u_dist(engine);
    rand_num[n_cnt] = u_dist(engine);
  }

  double buf = 0.0;
  for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
    buf += pow(rand_num[n_cnt], 2.0);
  }
  buf /= kNumData;

  for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
    rand_num[n_cnt] = rand_num[n_cnt] / sqrt(buf);
  }

  std::ofstream ofs_hist("histogram.dat", std::ios::out | std::ios::trunc);
  double max = *max_element(uniform_rand.begin(), uniform_rand.end());
  double min = *min_element(uniform_rand.begin(), uniform_rand.end());
  double data_length = max - min;
  int num_division = 100;
  std::vector<int> hist_data0(num_division, 0);

  for (int hist_x = 0; hist_x < num_division; ++hist_x) {
    for (int hist_y = 0; hist_y < kNumData; ++hist_y) {
      if ((min + ((data_length / num_division) * hist_x)) <=
              uniform_rand[hist_y] &&
          (min + ((data_length / num_division) * (hist_x + 1))) >
              uniform_rand[hist_y]) {
        hist_data0[hist_x] += 1;
      }
    }
  }

  for (int hist_x = 0; hist_x < num_division; ++hist_x) {
    ofs_hist << (min + ((data_length / num_division) * hist_x)) +
                    ((data_length / num_division) / 2.0) << " "
             << hist_data0[hist_x] << std::endl;
  }
}

void ICA::MakeSource(std::vector<double> &rand_num1,
                     std::vector<double> &rand_num2,
                     std::vector<double> &data_x1,
                     std::vector<double> &data_x2) {
  for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
    data_x1[n_cnt] = rand_num1[n_cnt] + 0.5 * rand_num2[n_cnt];
    data_x2[n_cnt] = 0.4 * rand_num1[n_cnt] + rand_num2[n_cnt];
  }
}

void ICA::EigenJacobiMethod(std::vector<std::vector<double> > &buf_matrix,
                            std::vector<std::vector<double> > &eigenvector) {
  std::vector<double> buf_matrix_y(kNumData, 0.0);
  std::vector<double> buf_matrix_x(kNumData, 0.0);

  for (int matrix_y = 0; matrix_y < kNumMatrix; ++matrix_y) {
    for (int matrix_x = 0; matrix_x < kNumMatrix; ++matrix_x) {
      eigenvector[matrix_y][matrix_x] = (matrix_y == matrix_x) ? 1.0 : 0.0;
    }
  }

  int cnt = 0;
  for (;;) {
    int matrix_y = 0, matrix_x = 0;

    double x_cnt = 0.0;
    for (int row_cnt = 0; row_cnt < kNumMatrix; ++row_cnt) {
      for (int column_cnt = 0; column_cnt < kNumMatrix; ++column_cnt) {
        if (row_cnt != column_cnt &&
            fabs(buf_matrix[row_cnt][column_cnt]) > x_cnt) {
          matrix_y = row_cnt;
          matrix_x = column_cnt;
          x_cnt = fabs(buf_matrix[row_cnt][column_cnt]);
        }
      }
    }

    double row_row = buf_matrix[matrix_y][matrix_y];
    double col_col = buf_matrix[matrix_x][matrix_x];
    double row_col = buf_matrix[matrix_y][matrix_x];
    double alpha = (row_row - col_col) / 2.0;
    double beta = sqrt(alpha * alpha + row_col * row_col);
    double buf_cos = sqrt((1.0 + fabs(alpha) / beta) / 2.0);
    double buf_sin = (((row_row - col_col) >= 0.0) ? 1.0 : -1.0) * row_col /
                     (2.0 * beta * buf_cos);

    for (int m_cnt = 0; m_cnt < kNumMatrix; ++m_cnt) {
      if (m_cnt == matrix_y || m_cnt == matrix_x) continue;

      double matrix_row = buf_matrix[matrix_y][m_cnt];
      double matrix_col = buf_matrix[matrix_x][m_cnt];

      buf_matrix_y[m_cnt] = matrix_row * buf_cos + matrix_col * buf_sin;
      buf_matrix_x[m_cnt] = -matrix_row * buf_sin + matrix_col * buf_cos;
    }

    double buf_row_row = row_row * buf_cos * buf_cos +
                         2.0 * row_col * buf_cos * buf_sin +
                         col_col * buf_sin * buf_sin;
    double buf_row_col = 0.0;
    double buf_col_col = row_row * buf_sin * buf_sin -
                         2.0 * row_col * buf_cos * buf_sin +
                         col_col * buf_cos * buf_cos;
    double buf_col_row = 0.0;

    for (int m_cnt = 0; m_cnt < kNumMatrix; ++m_cnt) {
      buf_matrix[matrix_y][m_cnt] = buf_matrix[m_cnt][matrix_y] =
          buf_matrix_y[m_cnt];
      buf_matrix[matrix_x][m_cnt] = buf_matrix[m_cnt][matrix_x] =
          buf_matrix_x[m_cnt];
    }

    buf_matrix[matrix_y][matrix_y] = buf_row_row;
    buf_matrix[matrix_y][matrix_x] = buf_row_col;
    buf_matrix[matrix_x][matrix_x] = buf_col_col;
    buf_matrix[matrix_x][matrix_y] = buf_col_row;

    for (int m_cnt = 0; m_cnt < kNumMatrix; ++m_cnt) {
      double eigenvector_y = eigenvector[m_cnt][matrix_y];
      double eigenvector_x = eigenvector[m_cnt][matrix_x];

      buf_matrix_y[m_cnt] = eigenvector_y * buf_cos + eigenvector_x * buf_sin;
      buf_matrix_x[m_cnt] = -eigenvector_y * buf_sin + eigenvector_x * buf_cos;
    }
    for (int m_cnt = 0; m_cnt < kNumMatrix; ++m_cnt) {
      eigenvector[m_cnt][matrix_y] = buf_matrix_y[m_cnt];
      eigenvector[m_cnt][matrix_x] = buf_matrix_x[m_cnt];
    }

    double buf_error = 0.0;
    for (int column_cnt = 0; column_cnt < kNumMatrix; ++column_cnt) {
      for (int row_cnt = 0; row_cnt < kNumMatrix; ++row_cnt) {
        if (row_cnt != column_cnt) {
          buf_error += fabs(buf_matrix[column_cnt][row_cnt]);
        }
      }
    }

    if (buf_error < kConvergenceError) break;
    ++cnt;
    if (cnt > kMaxNumIterations) break;
  }
}

double ICA::H4(double &y) { return pow(y, 4.0) - (6.0 * pow(y, 2.0)) + 3.0; }

double ICA::H3(double &y) { return 4.0 * pow(y, 3.0) - (3.0 * y); }

void ICA::GradientAlgorithm(std::vector<double> &z1, std::vector<double> &z2) {
  std::vector<double> y1(kNumData, 0.0);
  std::vector<double> y2(kNumData, 0.0);
  std::vector<double> alpha(kNumK, 0.0);
  double alpha_init = 90.0;

  alpha[0] = alpha_init;
  std::stringstream alpha_file_name;
  alpha_file_name.str("");
  alpha_file_name << "alpha" << static_cast<int>(fabs(alpha_init)) << ".dat"
                  << std::flush;
  std::ofstream ofs_alpha(alpha_file_name.str().c_str(),
                          std::ios::out | std::ios::trunc);

  for (int k_cnt = 0; k_cnt < kNumK - 1; ++k_cnt) {
    double radian = alpha[k_cnt] * (M_PI / 180);

    for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
      y1[n_cnt] = (cos(radian) * z1[n_cnt]) + (-sin(radian) * z2[n_cnt]);
      y2[n_cnt] = (sin(radian) * z1[n_cnt]) + (cos(radian) * z2[n_cnt]);
    }

    if ((k_cnt + 1) % 10 == 0 && alpha[k_cnt] > 0) {
      std::stringstream file_name;
      file_name.str("");
      file_name << static_cast<int>(fabs(alpha_init)) << "y_data"
                << static_cast<int>(fabs(alpha[k_cnt])) << ".dat" << std::flush;
      std::ofstream ofs_y(file_name.str().c_str(),
                          std::ios::out | std::ios::trunc);

      for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
        ofs_y << y1[n_cnt] << " " << y2[n_cnt] << std::endl;
      }
    }

    double h_buf1 = 0.0, h_buf2 = 0.0;

    for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
      h_buf1 += pow(y1[n_cnt], 4.0) - 6.0 * pow(y1[n_cnt], 2.0) + 3.0;
      h_buf2 += pow(y2[n_cnt], 4.0) - 6.0 * pow(y2[n_cnt], 2.0) + 3.0;
    }

    h_buf1 /= kNumData;
    h_buf2 /= kNumData;
    h_buf1 = 0.0;
    h_buf2 = 0.0;
    double h4_buf1 = 0.0, h3_buf1 = 0.0, h4_buf2 = 0.0, h3_buf2 = 0.0;

    for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
      h4_buf1 += H4(y1[n_cnt]);
      h3_buf1 += H3(y1[n_cnt]) * (-y2[n_cnt]);
    }

    for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
      h4_buf2 += H4(y2[n_cnt]);
      h3_buf2 += H3(y2[n_cnt]) * y1[n_cnt];
    }

    h4_buf1 /= kNumData, h3_buf1 /= kNumData, h4_buf2 /= kNumData,
        h3_buf2 /= kNumData;

    h_buf1 = h4_buf1 * h3_buf1;
    h_buf2 = h4_buf2 * h3_buf2;

    alpha[k_cnt + 1] = alpha[k_cnt] + kMu * (h_buf1 + h_buf2);
    ofs_alpha << k_cnt << " " << alpha[k_cnt + 1] << std::endl;
    std::cout << alpha[k_cnt + 1] << std::endl;
  }
}

void ICA::Calculation() {
  std::vector<double> rand_num_s1(kNumData, 0.0);
  std::vector<double> rand_num_s2(kNumData, 0.0);
  std::vector<double> x1(kNumData, 0.0);
  std::vector<double> x2(kNumData, 0.0);
  std::vector<std::vector<double> > matrix1(
      kNumMatrix, std::vector<double>(kNumMatrix, 0.0));
  std::vector<std::vector<double> > matrix2(
      kNumMatrix, std::vector<double>(kNumMatrix, 0.0));
  std::vector<std::vector<double> > buf_matrix(
      kNumMatrix, std::vector<double>(kNumMatrix, 0.0));
  std::vector<double> z1(kNumData, 0.0);
  std::vector<double> z2(kNumData, 0.0);

  GenerateRandNum(rand_num_s1);
  GenerateRandNum(rand_num_s2);
  MakeSource(rand_num_s1, rand_num_s2, x1, x2);

  std::ofstream ofs_data("s_x_data.dat", std::ios::out | std::ios::trunc);
  std::ofstream ofs_z("z_data.dat", std::ios::out | std::ios::trunc);
  for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
    ofs_data << rand_num_s1[n_cnt] << " " << rand_num_s2[n_cnt] << " "
             << x1[n_cnt] << " " << x2[n_cnt] << std::endl;
  }

  for (int matrix_y = 0; matrix_y < kNumMatrix; ++matrix_y) {
    for (int matrix_x = 0; matrix_x < kNumMatrix; ++matrix_x) {
      for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
        if (matrix_y == 0 && matrix_x == 1) {
          matrix1[matrix_y][matrix_x] += x1[n_cnt] * x2[n_cnt];
        }
        if (matrix_y == 1 && matrix_x == 0) {
          matrix1[matrix_y][matrix_x] += x1[n_cnt] * x2[n_cnt];
        }
        if (matrix_y == 0 && matrix_x == 0) {
          matrix1[matrix_y][matrix_x] += pow(x1[n_cnt], 2.0);
        }
        if (matrix_y == 1 && matrix_x == 1) {
          matrix1[matrix_y][matrix_x] += pow(x2[n_cnt], 2.0);
        }
      }
      matrix1[matrix_y][matrix_x] /= kNumData;
    }
  }

  for (int matrix_y = 0; matrix_y < kNumMatrix; ++matrix_y) {
    for (int matrix_x = 0; matrix_x < kNumMatrix; ++matrix_x) {
      std::cout << matrix1[matrix_y][matrix_x] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  EigenJacobiMethod(matrix1, matrix2);

  for (int matrix_y = 0; matrix_y < kNumMatrix; ++matrix_y) {
    for (int matrix_x = 0; matrix_x < kNumMatrix; ++matrix_x) {
      std::cout << matrix1[matrix_y][matrix_x] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  for (int matrix_y = 0; matrix_y < kNumMatrix; ++matrix_y) {
    for (int matrix_x = 0; matrix_x < kNumMatrix; ++matrix_x) {
      std::cout << matrix2[matrix_y][matrix_x] << " ";
    }
    std::cout << std::endl;
  }

  matrix1[0][0] = 1.0 / sqrt(matrix1[0][0]);
  matrix1[1][1] = 1.0 / sqrt(matrix1[1][1]);

  for (int matrix_y = 0; matrix_y < kNumMatrix; ++matrix_y) {
    for (int matrix_x = 0; matrix_x < kNumMatrix; ++matrix_x) {
      for (int matrix_z = 0; matrix_z < kNumMatrix; ++matrix_z) {
        buf_matrix[matrix_y][matrix_x] +=
            (matrix1[matrix_y][matrix_z] * matrix2[matrix_x][matrix_z]);
      }
    }
  }

  std::cout << std::endl;

  for (int matrix_y = 0; matrix_y < kNumMatrix; ++matrix_y) {
    for (int matrix_x = 0; matrix_x < kNumMatrix; ++matrix_x) {
      std::cout << buf_matrix[matrix_y][matrix_x] << " ";
    }
    std::cout << std::endl;
  }

  for (int n_cnt = 0; n_cnt < kNumData; ++n_cnt) {
    z1[n_cnt] = (buf_matrix[0][0] * x1[n_cnt]) + (buf_matrix[0][1] * x2[n_cnt]);
    z2[n_cnt] = (buf_matrix[1][0] * x1[n_cnt]) + (buf_matrix[1][1] * x2[n_cnt]);
    ofs_z << z1[n_cnt] << " " << z2[n_cnt] << std::endl;
  }

  GradientAlgorithm(z1, z2);
}

int main(void) {
  ICA obj_ica;

  obj_ica.Calculation();

  return 0;
}
