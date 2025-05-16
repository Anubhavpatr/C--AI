#ifndef MATRIX_H
#define MATRIX_H
// #include <thread>
// #include <mutex>
#include <iostream>
#include <tuple>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <exception>
#include "Tensor.h"
#include "Logger.h"

template <typename T>
class ThreeDArray;

template<typename T>
class Matrix {
    int rows;
    int columns;
    int size; // for vector
    std::shared_ptr<T[]> data;
    std::vector<int> shape_;

    Matrix() = default;

public:

    Matrix(int rows, int columns, T fill_value)
        : rows(rows), columns(columns),
         data(std::shared_ptr<T[]>(new T[rows * columns], std::default_delete<T[]>())),
         shape_({rows,columns}) {
        for (int i = 0; i < rows * columns; i++) {
            data[i] = fill_value;
        }
    }

    Matrix(int rows,int columns) :  rows(rows), columns(columns), data(nullptr),shape_({rows,columns}),size(-1) {}

    Matrix(int rows, int columns, std::shared_ptr<T[]> data)
        : rows(rows), columns(columns), data(data),shape_({rows,columns}),size(-1) {}

    Matrix(int size,std::shared_ptr<T[]> data)
        : rows(-1),columns(-1),size(size),shape_({-1,-1}),data(data) {}

    Matrix(const Matrix& matrix)
        : rows(matrix.rows), columns(matrix.columns), data(matrix.data),size(-1) {}

    // could have used template<typename ...Args>
    // but if i pass a single argument could cause ambiguity

    T& operator[](std::tuple<int,int> position){
        int index = std::get<0>(position) * columns + std::get<1>(position);
        try
        {
            if(index >= rows * columns)
            {
                throw std::runtime_error("Wrong index not accessible");
            }
            Logger::info("Succesfully accessed the element");
        }
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what()));
            std::cerr << e.what() << std::endl;
        }
        catch(...)
        {
            Logger::error("Error while accessing the element");
            std::cerr << "Error while accessing the element" << std::endl;
        }
        return data[index];
    }

    std::vector<T> operator[](std::tuple<int> position) {
        std::vector<T> row;
        int offset = std::get<0>(position) * columns;
        for (int i = 0; i < columns; ++i) {
            // making a shared pointer for each and every tensor
            row.push_back(data[offset + i]);
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
        try
        {
            if(rows != -1 && columns != -1)
            {
                if ((rows != target_rows && rows != 1) ||
                    (columns != target_cols && columns != 1)) {
                    throw std::runtime_error("Incompatible shapes for broadcasting");
                }
            }
        }
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what()));
            std::cerr << e.what() << std::endl;
        }
        // convert the vector to a matrix in case of broadcasting it
        if(rows == -1 && columns == -1)
        {
            this->rows = 1;
            this->columns = this->size;
            this->size = -1;
        }

        Matrix<T> result{target_rows, target_cols};
        result.data = std::shared_ptr<T[]>(new T[target_rows * target_cols]);

        try
        {
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
        }
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what()));
            std::cerr << e.what() << std::endl;
        }
        catch(...)
        {
            Logger::error("Error while broadcasting");
            std::cerr << "Error while broadcasting" << std::endl;
        }

        return result;
    }

    template<typename _T>
    typename std::enable_if<
        std::is_same_v<std::decay_t<_T>,Matrix<T>>,
        Matrix<T>
    >::type
    static ones_like(_T&& a)
    {
        // goes out of scope so therefore did not use try and catch
        Matrix<T> result{a.rows,a.columns,1.0};
        return result;
    }

    template<typename _T>
    typename std::enable_if<
        std::is_same_v<std::decay_t<_T>,Matrix<T>>,
        Matrix<T>
    >::type
    static zeros_like(_T&& a)
    {
        // goes out of scope so therefore did not use try and catch
        Matrix<T> result{a.rows,a.columns,0.0};
        return result;
    }

    Matrix<T> max(int dim=0,bool keepdim=false)
    {
        Matrix<T> result{};
        try
        {
            if(dim == 0)
            {
                if(keepdim)
                {
                    std::shared_ptr<T[]> new_data(new T[1 * columns]);
                    for(int i = 0;i < columns;i++)
                    {
                        new_data[i] = T{};
                        for(int j = 0;j < rows;j++)
                        {
                            if((new_data[i] < data[j * columns+i]).value())
                            {
                                new_data[i] = data[j * columns+i];
                            }
                        }
                    }
                    result = Matrix<T>(1,columns,new_data);
                }
                else
                {
                    std::shared_ptr<T[]> new_data(new T[1 * columns]);
                    for(int i = 0;i < columns;i++)
                    {
                        new_data[i] = T{};
                        for(int j = 0;j < rows;j++)
                        {
                            if((new_data[i] < data[j * columns + i]).value())
                            {
                                new_data[i] = data[j * columns + i];
                            }
                        }
                    }

                    result = Matrix<T>{columns,new_data};
                }
            }
            else if(dim == 1 || dim == -1)
            {
                if(keepdim)
                {
                    std::shared_ptr<T[]> new_data(new T[rows * 1]);
                    for(int i = 0;i < rows;i++)
                    {
                        new_data[i] = T{};
                        for(int j = 0;j < columns;j++)
                        {
                            if((new_data[i] < this->data[i * columns + j]).value())
                            {
                                new_data[i] = this->data[i * columns + j];
                            }
                        }
                    }

                    result = Matrix<T>{rows,1,new_data};
                }
                else
                {
                    std::shared_ptr<T[]> new_data(new T[rows * 1]);
                    for(int i = 0;i < rows;i++)
                    {
                        new_data[i] = T{};
                        for(int j = 0;j < columns;j++)
                        {
                            if((new_data[i] < this->data[i * columns + j]).value())
                            {
                                new_data[i] = this->data[i * columns + j];
                            }
                        }
                    }

                    result = Matrix<T>{rows,new_data};
                }
            }
            else
            {
                throw std::runtime_error("The dimension does not exist");
            }
        }
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what()));
            std::cerr << e.what() << std::endl;
        }
        catch(...)
        {
            Logger::error("Error while doing the max accross a certain dimension");
            std::cerr << "Error while doing the max accross a certain dimension" << std::endl;
        }

        return result;
    }

    Matrix<T> min(int dim=0,bool keepdim=false)
    {
        Matrix<T> result{};
        try
        {
            if(dim == 0)
            {
                if(keepdim)
                {
                    std::shared_ptr<T[]> new_data(new T[1 * columns]);
                    for(int i = 0;i < columns;i++)
                    {
                        new_data[i] = T{std::numeric_limits<float>::max()};
                        for(int j = 0;j < rows;j++)
                        {
                            if((new_data[i] > data[j * columns+i]).value())
                            {
                                new_data[i] = data[j * columns+i];
                            }
                        }
                    }
                    result = Matrix<T>(1,columns,new_data);
                }
                else
                {
                    std::shared_ptr<T[]> new_data(new T[1 * columns]);
                    for(int i = 0;i < columns;i++)
                    {
                        new_data[i] = T{std::numeric_limits<float>::max()};
                        for(int j = 0;j < rows;j++)
                        {
                            if((new_data[i] > data[j * columns + i]).value())
                            {
                                new_data[i] = data[j * columns + i];
                            }
                        }
                    }

                    result = Matrix<T>{columns,new_data};
                }
            }
            else if(dim == 1 || dim == -1)
            {
                if(keepdim)
                {
                    std::shared_ptr<T[]> new_data(new T[rows * 1]);
                    for(int i = 0;i < rows;i++)
                    {
                        new_data[i] = T{std::numeric_limits<float>::max()};
                        for(int j = 0;j < columns;j++)
                        {
                            if((new_data[i] > this->data[i * columns + j]).value())
                            {
                                new_data[i] = this->data[i * columns + j];
                            }
                        }
                    }

                    result = Matrix<T>{rows,1,new_data};
                }
                else
                {
                    std::shared_ptr<T[]> new_data(new T[rows * 1]);
                    for(int i = 0;i < rows;i++)
                    {
                        new_data[i] = T{std::numeric_limits<float>::max()};
                        for(int j = 0;j < columns;j++)
                        {
                            if((new_data[i] > this->data[i * columns + j]).value())
                            {
                                new_data[i] = this->data[i * columns + j];
                            }
                        }
                    }

                    result = Matrix<T>{rows,new_data};
                }
            }
            else
            {
                throw std::runtime_error("The dimension does not exist");
            }
        }
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what()));
            std::cerr << e.what() << std::endl;
        }
        catch(...)
        {
            Logger::error("Error while doing the min accross a certain dimension");
            std::cerr << "Error while doing the min accross a certain dimension" << std::endl;
        }

        return result;
    }

    T sum()
    {
        T sum = T{};
        try
        {
            if(rows != -1 && columns != -1)
            {
                throw std::runtime_error("Only Vectors are allowed not 2D Matrix");
            }
            
            if(this->size == -1)
            {
                throw std::runtime_error("Only Vectors are allowed");
            }

            for(int i = 0;i < this->size;i++)
            {
                sum += this->data[i];
            }

        }
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what()));
            std::cerr << e.what() << std::endl;
        }
        catch(...)
        {
            Logger::error("Error while doing sum of a vector");
            std::cerr << "Error while doing sum of a vector" << std::endl;
        }

        return sum;
    }


    Matrix<T> clone()
    {
        Matrix<T> result{this->rows,this->columns};
        try
        {
            result.data = std::shared_ptr<T[]>(new T[rows * columns]);
            for(int i = 0;i < rows * columns ;i++)
            {
                result.data[i] = T{this->data[i].value()};
            }
        }
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what()));
            std::cerr << e.what() << std::endl;
        }
        catch(...)
        {
            Logger::error("Error while cloning the matrix");
            std::cerr << "Error while clonning the matrix" << std::endl;
        }
        return result;
    }


    Matrix<T> sum(int dim,bool keepdim)
    {
        Matrix<T> result{};
        try
        {
            if(dim == 0)
            {
                if (keepdim)
                {
                    std::shared_ptr<T[]> new_data(new T[1 * columns]);
                    for(int i = 0;i < columns;i++)
                    {
                        new_data[i] = T{};
                        for(int j = 0;j < rows;j++)
                        {
                            new_data[i] += data[j * columns+i];
                        }
                    }
                    result = Matrix<T>(1,columns,new_data);
                }
                else
                {
                    //std::cout << "I am here" << std::endl;
                    std::shared_ptr<T[]> new_data(new T[1 * columns]);
                    for(int i = 0;i < columns;i++)
                    {
                        new_data[i] = T{};
                        for(int j = 0;j < rows;j++)
                        {
                            new_data[i] += data[j * columns+i];
                        }
                    }
                    result = Matrix<T>(columns , new_data);
                }
            }
            else if(dim ==1 || dim == -1 )
            {
                if(keepdim)
                {
                    //std::cout << "I am here" << std::endl;
                    std::shared_ptr<T[]> new_data(new T[rows * 1]);
                    for(int i = 0;i < rows;i++)
                    {
                        new_data[i] = T{};
                        for(int j = 0;j < columns;j++)
                        {
                            new_data[i] += data[i * columns + j];
                        }
                    }
                    result = Matrix<T>(rows, 1, new_data);
                }
                else
                {
                    std::shared_ptr<T[]> new_data(new T[rows * 1]);
                    for(int i = 0;i < rows;i++)
                    {
                        new_data[i] = T{};
                        for(int j = 0;j < columns;j++)
                        {
                            new_data[i] += data[i * columns + j];
                        }
                    }
                    result = Matrix<T>(rows, new_data);
                }
            }

            Logger::info("Successfully summed accross a dimension");
        }
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what())); // e.what() returns a const char*
            std::cerr << e.what() << std::endl;
        }
        catch(...)
        {
            Logger::error("Error while summing accross a dimension");
            std::cerr << "Error while summing accross a dimension" << std::endl;
        }
       //std::cout << "I am here" << std::endl;
        return result;
    }

    // i want both rvalue and lvalue to be passed
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
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what()));
            std::cerr << e.what() << std::endl;
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
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what()));
            std::cerr << e.what() << std::endl;
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
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what()));
            std::cerr << e.what() << std::endl;
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

    Matrix<T> pow(int num)
    {
        std::shared_ptr<T[]> new_data(new T[rows * columns]);
        for(int i = 0;i < rows * columns;i++)
        {
            new_data[i] = this->data[i].pow(num);
        }
        return {rows,columns,new_data};
    }



    std::string shape() const {
        return "(" + std::to_string(rows) + ", " + std::to_string(columns) + ")";
    }

    int shape(int index)
    {
        try
        {
            if(index > 2 || index < 0)
            {
                throw std::runtime_error("accessing the wrong index");
            }
            Logger::info("Succesfully will access the index");
        }
        catch(const std::exception& e)
        {
            Logger::error(std::string(e.what()));
            std::cerr << e.what() << std::endl;
        }
        catch(...)
        {
            Logger::error("Error while accessing the shape");
            std::cerr << "Error while accessing the shape" << std::endl;
        }
        return this->shape_[index];
    } 

    void print() const {
        if(rows > 0 && columns > 0)
        {
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
        else if(rows == -1 && columns == -1)
        {
            std::cout << "[";
            for(int i = 0;i < this->size;i++)
            {
                std::cout << data[i];
                if(i != this->size - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
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