#ifndef MATRIX_H
#define MATRIX_H
// #include <thread>
// #include <mutex>
#include <iostream>
#include <tuple>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "Logger.h"

template <typename T>
class ThreeDArray;

template<typename T>
class Matrix {
    int rows;
    int columns;
    std::shared_ptr<T[]> data;

public:

    Matrix(int rows, int columns, T fill_value)
        : rows(rows), columns(columns), data(std::shared_ptr<T[]>(new T[rows * columns], std::default_delete<T[]>())) {
        for (int i = 0; i < rows * columns; i++) {
            data[i] = fill_value;
        }
    }

    Matrix(int rows,int columns) :  rows(rows), columns(columns), data(nullptr) {}

    Matrix(int rows, int columns, std::shared_ptr<T[]> data)
        : rows(rows), columns(columns), data(data) {}

    Matrix(const Matrix& matrix)
        : rows(matrix.rows), columns(matrix.columns), data(matrix.data) {}

    // could have used template<typename ...Args>
    // but if i pass a single argument could cause ambiguity

    T& operator[](std::tuple<int,int> position){
        int index = std::get<0>(position) * columns + std::get<1>(position);
        return data[index];
    }

    std::vector<std::shared_ptr<T>> operator[](std::tuple<int> position) {
        std::vector<std::shared_ptr<T>> row;
        int offset = std::get<0>(position) * columns;
        for (int i = 0; i < columns; ++i) {
            // making a shared pointer for each and every tensor
            row.push_back(std::make_shared<T>(data[offset + i]));
        }
        return row;
    }

    ThreeDArray<T> operator[](const std::vector<std::vector<int>>& indices) {
        std::vector<std::shared_ptr<T>> data_new;
        for (const auto& row : indices) {
            for (int idx : row) {
                // facing an ambiguity since both operators can take an initializer list
                // so i need to mention explicitely which method to call
                auto temp = (*this)[std::tuple<int>{idx}];
                data_new.insert(data_new.end(), temp.begin(), temp.end());
            }
        }
        return {static_cast<int>(indices.size()), static_cast<int>(indices[0].size()), columns, data_new};
    }

    Matrix<T> broadcast_to(int target_rows, int target_cols) {
        if ((rows != target_rows && rows != 1) ||
            (columns != target_cols && columns != 1)) {
            throw std::runtime_error("Incompatible shapes for broadcasting");
        }

        Matrix<T> result{target_rows, target_cols};
        result.data = std::shared_ptr<T[]>(new T[target_rows * target_cols]);

        for (int i = 0; i < target_rows; ++i) {
            for (int j = 0; j < target_cols; ++j) {
                int i_ = (rows == 1) ? 0 : i;
                int j_ = (columns == 1) ? 0 : j;

                // Compute the index in the original and broadcasted matrix
                int orig_idx = i_ * columns + j_;
                int target_idx = i * target_cols + j;

                // Key: preserve autodiff by copying references, not values
                result.data[target_idx] = data[orig_idx]; // Shallow copy of T
            }
        }

        return result;
    }

    
    template<typename _T>
    Matrix<T> matmul(_T&& a) {
        static_assert(std::is_same_v<std::decay_t<_T>, Matrix<T>>, "Invalid argument type");
        std::shared_ptr<T[]> new_data(new T[rows * a.columns]);
        try
        {
            for (int i = 0; i < rows * a.columns; i++) {
                new_data[i] = T{};
                for (int j = 0; j < columns; j++) {
                    new_data[i] += data[(i / a.columns) * columns + j] *
                                a.data[i % a.columns + j * a.columns];
                }
            }
            Logger::info("Successfully matrix multiplied two matrices");
        }
        catch(...)
        {
            Logger::error("Error while matrix multiplying two matrices");
            std::cerr << "Error while matrix multiplying two matrices" << std::endl;
        }
        return {rows, a.columns, new_data};
    }

    //  Broadcasting needs to be implemented
    // creates an ambiguity
    // if it is not fully specialized
    template<typename _T>  
    typename std::enable_if<   
        std::is_same_v<std::decay_t<_T>,Matrix<T>>,
        Matrix<T>
    >::type
    operator+(_T&& a) {
        //ownership will be shared so it will not call the destructor
        // which atleast someone owns it
        int out_rows = std::max(rows, a.rows);
        int out_cols = std::max(columns, a.columns);

        // Matrix<T> result{out_rows, out_cols,T{}};
        std::shared_ptr<T[]> new_data(new T[out_rows * out_cols]);
        try
        {            
            Matrix<T> A = this->broadcast_to(out_rows, out_cols);
            Matrix<T> B = a.broadcast_to(out_rows, out_cols);
            for (int i = 0; i < out_rows; ++i)
            {
                for (int j = 0; j < out_cols; ++j)
                {
                    new_data[i * out_cols + j] = A[{i, j}] + B[{i,j}];
                }
            }

            Logger::info("Successfully calculated element wise operation of + using broadcasting");
        }
        catch(...)
        {
            Logger::error("Error while calculating element wise operation of + using broadcasting");
            std::cerr << "Error while calculating element wise operation of + using broadcasting" << std::endl;
        }

        return {out_rows,out_cols,new_data};
    }
    
    template<typename _T>  
    typename std::enable_if<   
        std::is_same_v<std::decay_t<_T>,Matrix<T>>,
        Matrix<T>
    >::type
    operator-(_T&& a) {
        //ownership will be shared so it will not call the destructor
        // which atleast someone owns it
        int out_rows = std::max(rows, a.rows);
        int out_cols = std::max(columns, a.columns);

        // Matrix<T> result{out_rows, out_cols,T{}};
        std::shared_ptr<T[]> new_data(new T[out_rows * out_cols]);
        try
        {            
            Matrix<T> A = this->broadcast_to(out_rows, out_cols);
            Matrix<T> B = a.broadcast_to(out_rows, out_cols);
            for (int i = 0; i < out_rows; ++i)
            {
                for (int j = 0; j < out_cols; ++j)
                {
                    new_data[i * out_cols + j] = A[{i,j}] - B[{i,j}];
                }
            }

            Logger::info("Successfully calculated element wise operation of - using broadcasting");
        }
        catch(...)
        {
            Logger::error("Error while calculating element wise operation of - using broadcasting");
            std::cerr << "Error while calculating element wise operation of - using broadcasting" << std::endl;
        }

        return {out_rows,out_cols,new_data};
    }

    template<typename _T>
    std::enable_if<
        std::is_same_v<std::decay_t<_T>,Matrix<T>>,
        Matrix<T>
    >::type
    operator*(_T&& a)
    {
        //ownership will be shared so it will not call the destructor
        // which atleast someone owns it
        int out_rows = std::max(rows, a.rows);
        int out_cols = std::max(columns, a.columns);

        // Matrix<T> result{out_rows, out_cols,T{}};
        std::shared_ptr<T[]> new_data(new T[out_rows * out_cols]);
        try
        {            
            Matrix<T> A = this->broadcast_to(out_rows, out_cols);
            Matrix<T> B = a.broadcast_to(out_rows, out_cols);
            for (int i = 0; i < out_rows; ++i)
            {
                for (int j = 0; j < out_cols; ++j)
                {
                    new_data[i * out_cols + j] = A[{i,j}] * B[{i,j}];
                }
            }

            Logger::info("Successfully calculated element wise operation of * using broadcasting");
        }
        catch(...)
        {
            Logger::error("Error while calculating element wise operation of * using broadcasting");
            std::cerr << "Error while calculating element wise operation of * using broadcasting" << std::endl;
        }

        return {out_rows,out_cols,new_data};
    }

    template<typename _T>
    std::enable_if<
        std::is_same_v<std::decay_t<_T>,Matrix<T>>,
        Matrix<T>
    >::type
    operator/(_T&& a)
    {
        //ownership will be shared so it will not call the destructor
        // which atleast someone owns it
        int out_rows = std::max(rows, a.rows);
        int out_cols = std::max(columns, a.columns);

        // Matrix<T> result{out_rows, out_cols,T{}};
        std::shared_ptr<T[]> new_data(new T[out_rows * out_cols]);
        try
        {            
            Matrix<T> A = this->broadcast_to(out_rows, out_cols);
            Matrix<T> B = a.broadcast_to(out_rows, out_cols);
            for (int i = 0; i < out_rows; ++i)
            {
                for (int j = 0; j < out_cols; ++j)
                {
                    new_data[i * out_cols + j] = A[{i,j}] / B[{i,j}];
                }
            }

            Logger::info("Successfully calculated element wise operation of / using broadcasting");
        }
        catch(...)
        {
            Logger::error("Error while calculating element wise operation of / using broadcasting");
            std::cerr << "Error while calculating element wise operation of / using broadcasting" << std::endl;
        }

        return {out_rows,out_cols,new_data};
    }   

    Matrix<T> view(std::tuple<int, int> size) {
        int total = std::get<0>(size) * std::get<1>(size);
        if (total != rows * columns) {
            throw std::runtime_error("Incorrect shape.");
        }
        return {std::get<0>(size), std::get<1>(size), data};
    }

    ThreeDArray<T> view(std::tuple<int, int, int> size) {
        int total = std::get<0>(size) * std::get<1>(size) * std::get<2>(size);
        if (total != rows * columns) {
            throw std::runtime_error("Incorrect shape.");
        }
        std::vector<std::shared_ptr<T>> data_vec;
        for (int i = 0; i < total; i++) {
            data_vec.push_back(std::make_shared<T>(data[i]));
        }
        return {std::get<0>(size), std::get<1>(size), std::get<2>(size), data_vec};
    }

    Matrix<T> transpose() {
        std::shared_ptr<T[]> new_data(new T[rows * columns]);
        int index = 0;
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                new_data[index++] = data[j * columns + i];
            }
        }
        return {columns, rows, new_data};
    } 

    std::string shape() const {
        return "(" + std::to_string(rows) + ", " + std::to_string(columns) + ")";
    }

    void print() const {
        std::cout << "[\n";
        for (int i = 0; i < rows; ++i) {
            std::cout << "[";
            for (int j = 0; j < columns; ++j) {
                std::cout << data[i * columns + j];
                if (j != columns - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        std::cout << "]\n";
    }
};

template <typename T>
class ThreeDArray {
public:
    int batch_size;
    int context_size;
    int embedding_dim;
    std::vector<std::shared_ptr<T>> data;

    ThreeDArray(int batch_size, int context_size, int embedding_dim,
                std::vector<std::shared_ptr<T>> data)
        : batch_size(batch_size),
          context_size(context_size),
          embedding_dim(embedding_dim),
          data(std::move(data)) {}

    Matrix<T> view(int first_dim, int second_dim) {
        int total_elements = batch_size * context_size * embedding_dim;
        if (second_dim == 1) {
            second_dim = total_elements / first_dim;
        }

        auto new_data = std::make_shared<T[]>(total_elements);
        for (int i = 0; i < total_elements; ++i) {
            new_data[i] = *data[i];
        }

        return {first_dim, second_dim, new_data};
    }

    std::vector<std::shared_ptr<T>> operator[](std::tuple<int, int> position) {
        int base = std::get<0>(position) * context_size * embedding_dim +
                   std::get<1>(position) * embedding_dim;
        return {data.begin() + base, data.begin() + base + embedding_dim};
    }


    void print() const {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < context_size; ++j) {
                for (int k = 0; k < embedding_dim; ++k) {
                    int idx = i * context_size * embedding_dim + j * embedding_dim + k;
                    std::cout << *data[idx] << " ";
                }
                std::cout << "\n";
            }
        }
    }
};

#endif