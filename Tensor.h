#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <type_traits>
#include <cmath>
#include <functional>
#include "Logger.h"

// 3. Logical (for boolean arrays)
// & (logical AND)

// | (logical OR)

// ^ (logical XOR)

// ~ (logical NOT — unary, but can apply per element)

// std::enable if used since template same number of parameters
// it causes ambiguity for the compiler 
// that which template are we trying to call

// Here Pointer to Implementation is required

// if using template functions then declare and define in 

class Impl {
public:
    float val;
    float grad = 0.0f;
    std::function<void()> backward_fn;
    std::vector<std::shared_ptr<Impl>> prev;

    Impl(float v) : val(v) {}
};
// this is required since this Impl is given to be owned by someone
//atleast but i do not want (this) pointer to  be owned by someone
// everytime
template<typename>
inline constexpr bool always_false = false;

class Tensor {
    std::shared_ptr<Impl> impl; // if this shared pointer has no owner then it
    //  going to get destroyed

public:
    Tensor(float val);
    Tensor();  // Default constructor
        // Shallow copy constructor
    Tensor(const Tensor& other) : impl(other.impl) {}

    // Shallow copy assignment
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            impl = other.impl;
        }
        return *this;
    }

    float value() const;
    float grad() const;
    void backward();

    template<typename T>
    Tensor operator+(T&& rhs)
    {
        Tensor out{};
        try
        {
            if constexpr(std::is_same_v<std::decay_t<T>,Tensor>)
            {            // Tensors can be lost but the actual content 
                // of the tensor needs to be shared
                out.impl->val = this->value() +  rhs.value();
                out.impl->prev = {this->impl,rhs.impl}; // transfering the ownership
                // of the actual content to a data structure
                // since we take the copy of the *this pointer 
                // so we do not access the updated gradients by previous backprop
                // which is flowing backwards
                out.impl->backward_fn = [object = *this, rhs_impl = rhs.impl, out_impl = out.impl]() {
                    object.impl->grad += out_impl->grad;
                    rhs_impl->grad += out_impl->grad;
                };

                Logger::info("Successfully added the tensor with another tensor");
            }
            else if constexpr(std::is_arithmetic_v<std::decay_t<T>>)
            {
                out.impl->val = this->value() + rhs;
                out.impl->prev = {this->impl}; // transfering the ownership
                out.impl->backward_fn = [object = *this, out_impl = out.impl]() {
                    object.impl->grad += out_impl->grad;
                };
                Logger::info("Successfully added the tensor with another number");
            }
            else
            {
                static_assert(always_false<T>, "Unsupported type for Tensor addition");
            }
        }
        catch(...)
        {
            Logger::error("Error in operator+ while adding two tensors OR adding a tensor wir=th a number");
            std::cerr << "Error in operator+ while adding two tensors" << std::endl;
        }
        return out;
    }


    template<typename T>
    Tensor operator-(T&& rhs)
    {
        Tensor out{};
        try
        {
            if constexpr(std::is_same_v<std::decay_t<T>,Tensor>)
            // Tensors can be lost but the actual content 
            // of the tensor needs to be shared
            {
                out.impl->val = this->value() -  rhs.value();
                out.impl->prev = {this->impl,rhs.impl}; // transfering the ownership
                out.impl->backward_fn = [object = *this, rhs_impl = rhs.impl, out_impl = out.impl]() {
                    object.impl->grad += out_impl->grad;
                    rhs_impl->grad -= out_impl->grad;
                };
                Logger::info("successfully substracted a tensor from a tensor");
            }
            else if constexpr(std::is_arithmetic_v<std::decay_t<T>>)
            {
                out.impl->val = this->value() -  rhs;
                out.impl->prev = {this->impl}; // transfering the ownership
                out.impl->backward_fn = [object = *this, out_impl = out.impl]() {
                    object.impl->grad += out_impl->grad;
                };
                Logger::info("successfully substracted a number from a tensor");
            }
            else
            {
                static_assert(always_false<T>, "Unsupported type for Tensor subtraction");
            }
        }
        catch(...)
        {
            Logger::error("Error while subtracting a tensor or number from a tensor");
            std::cerr << "Error while subtracting a tensor or number from a tensor" << std::endl;
        }
        return out;
    }


    template<typename T>
    Tensor operator*(T&& other) {
        Tensor out{};
        try {
            if constexpr (std::is_same_v<std::decay_t<T>, Tensor>) {
                // Case 1: Multiply by another Tensor
                out.impl->val = this->value() * other.value();
                out.impl->prev = {this->impl, other.impl};
                out.impl->backward_fn = [object = *this, other_impl = other.impl, out_impl = out.impl]() {
                    object.impl->grad += other_impl->val * out_impl->grad;
                    other_impl->grad += object.value() * out_impl->grad;
                };
                Logger::info("Successfully multiplied a tensor with another tensor");

            } else if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
                // Case 2: Multiply by a number
                out.impl->val = this->value() * other;
                out.impl->prev = {this->impl};
                out.impl->backward_fn = [object = *this, number = other, out_impl = out.impl]() {
                    object.impl->grad += number * out_impl->grad;
                };
                Logger::info("Successfully multiplied a tensor with a number");

            } else {
                static_assert(always_false<T>, "Unsupported type for Tensor multiplication");
            }
        } catch (...) {
            Logger::error("Error while multiplying a tensor");
            std::cerr << "Error while multiplying a tensor" << std::endl;
        }

        return out;
    }


    template<typename T>
    typename std::enable_if<
        std::is_same_v<std::decay_t<T>,Tensor>,
        Tensor
    >::type
    operator/(T&& other)
    {
        Tensor out{};
        try
        {   
            if constexpr(std::is_same_v<std::decay_t<T>,Tensor>)
            {
                out.impl->val = this->value() / other.value();
                out.impl->prev = {this->impl,other.impl};
                out.impl->backward_fn = [object = *this,&out,other_impl = other.impl](){
                    object.impl->grad += out.grad() / other_impl->val;
                    other_impl->grad += -1  * out.grad() / std::pow(other_impl->val,2);
                };

                Logger::info("Successfully divided a tensor by a tensor");
            }
            else if constexpr(std::is_arithmetic_v<std::decay_t<T>>)
            {
                out.impl->val = this->value() / other;
                out.impl->prev = {this->impl};
                out.impl->backward_fn = [object = *this,&out,number = other]()
                {
                    object.impl->grad += out.impl->grad / number;
                };
            Logger::info("Successfully divided a tensor by a number");
            }
            else
            {
                static_assert(always_false<T>, "Unsupported type for Tensor division");
            }
        }
        catch(...)
        {
            Logger::error("Error while divinding a tensor with a tensor");
            std::cerr << "Error while divinding a tensor with a tensor" << std::endl;
        }
        return out;
    }

    template<typename T>
    typename std::enable_if<
        std::is_same_v<std::decay_t<T>,Tensor> || std::is_arithmetic_v<std::decay_t<T>>,
        Tensor&
    >::type
    operator+=(T&& rhs)
    {
        try
        {
            // Tensors can be lost but the actual content 
            // of the tensor needs to be shared
            // this->impl->val = this->value() +  rhs.value();
            // this->impl->prev.push_back(rhs.impl); // transfering the ownership
            // of the actual content to a data structure
            
            if constexpr (std::is_rvalue_reference_v<T&&>)
            {
                *this = *this + std::move(rhs);
            }
            else
            {
                *this = *this + rhs;
            }


            // problem the same gradient is getting used to backprop where 
            // the certain tensor is not updated 
            // once created , then it is there in the graph
            // since i do not have the previous gradient since the previous gradient
            // is overwritten by the new gradient calculated


            // std::function<void()> prev_backward = this->impl->backward_fn;
            // this->impl->backward_fn = [this, rhs_impl = rhs.impl,prev_backward]() {
            //     if(prev_backward) prev_backward();
            //     this->impl->grad += this->impl->grad;
            //     rhs_impl->grad += this->impl->grad;
            // };
            Logger::info("Successfully added a tensor with itself");
        }
        catch(...)
        {
            Logger::error("Error while adding a tensor with itself");
            std::cerr << "Error in operator+" << std::endl;
        }
        return *this;
    }

    template<typename T>
    typename std::enable_if<
        std::is_same_v<std::decay_t<T>,Tensor> || std::is_arithmetic_v<std::decay_t<T>>,
        Tensor&
    >::type
    operator-=(T&& rhs)
    {
        try
        {
            if(std::is_rvalue_reference_v<T&&>)
            {
                *this = *this - std::move(rhs);
            }
            else
            {
                *this = *this - rhs;
            }
            Logger::info("Succesfully substracted a tensor from a tensor");
        }
        catch(...)
        {
            Logger::error("Succesfully substracted a tensor from a tensor");
            std::cerr << "Error in operator-"  << std::endl;
        }

        return *this;
    }

    template<typename T>
    typename std::enable_if<
        std::is_same_v<std::decay_t<T>,Tensor> || std::is_arithmetic_v<std::decay_t<T>>,
        Tensor&
    >::type
    operator*=(T&& other)
    {   
        try
        {   
            if(std::is_rvalue_reference_v<T&&>)
            {
                *this = *this * std::move(other);
            }
            else
            {
                *this = *this * other;
            }
            Logger::info("Successfully with multiplying with a tensor with itself");
        }
        catch(...)
        {
            Logger::error("Error while multiplying a tensor with itself");
            std::cerr << "Error while multiplying a tensor with itself" << std::endl;
        }
        return *this;
    }

    template<typename T>
    typename std::enable_if<
        std::is_same_v<std::decay_t<T>,Tensor>,
        Tensor
    >::type
    operator/=(T&& other)
    {
        try
        {   
            if(std::is_rvalue_reference_v<T&&>)
            {
                *this = *this / std::move(other);
            }
            else
            {
                *this = *this / other;
            }
            Logger::info("Successfully with multiplying with a tensor with itself");
        }
        catch(...)
        {
            Logger::error("Error while multiplying a tensor with itself");
            std::cerr << "Error while multiplying a tensor with itself" << std::endl;
        }
        return *this;
    }

    template<typename T>    
    Tensor operator==(T&& other)
    {
        Tensor out{};
        try
        {
            if constexpr(std::is_same_v<std::decay_t<T>,Tensor>)
            {
                out.impl->val = (this->value() == other.value());
            }
            else if constexpr(std::is_arithmetic_v<std::decay_t<T>>)
            {
                out.impl->val = (this->value() == other);
            }
            else
            {
                static_assert(always_false<T>, "Unsupported type for Tensor comparison operator ==");
            }
            Logger::info("Successfully compared a tensor with a tensor or a number using == operator");
        }
        catch(...)
        {
            Logger::error("Error while a comparing a tensor with a tensor or a number using == operator");
            std::cerr << "Error while a comparing a tensor with a tensor or a number using == operator" << std::endl;
        }
        return out;
    }

    template<typename T>
    Tensor operator!=(T&& other)
    {
        Tensor out{};
        try
        {
            if constexpr(std::is_same_v<std::decay_t<T>,Tensor>)
            {
                out.impl->val = (this->value() != other.value());
            }
            else if constexpr(std::is_arithmetic_v<std::decay_t<T>>)
            {
                out.impl->val = (this->value() != other);
            }
            else
            {
                static_assert(always_false<T>, "Unsupported type for Tensor comparison operator !=");
            }
            Logger::info("Successfully compared a tensor with a tensor or a number using != operator");
        }
        catch(...)
        {
            Logger::error("Error while a comparing a tensor with a tensor or a number using != operator");
            std::cerr << "Error while a comparing a tensor with a tensor or a number using != operator" << std::endl;
        }
        return out;        
    }

    // < (less than)
    template<typename T>
    Tensor operator<(T&& other)
    {
        Tensor out{};
        try
        {
            if constexpr(std::is_same_v<std::decay_t<T>,Tensor>)
            {
                out.impl->val = (this->value() < other.value());
            }
            else if constexpr(std::is_arithmetic_v<std::decay_t<T>>)
            {
                out.impl->val = (this->value() < other);
            }
            else
            {
                static_assert(always_false<T>, "Unsupported type for Tensor comparison operator <");
            }
            Logger::info("Successfully compared a tensor with a tensor or a number using < operator");
        }
        catch(...)
        {
            Logger::error("Error while a comparing a tensor with a tensor or a number using < operator");
            std::cerr << "Error while a comparing a tensor with a tensor or a number using < operator" << std::endl;
        }
        return out;        
    }

    // <= (less than or equal)
    template<typename T>
    Tensor operator<=(T&& other)
    {
        Tensor out{};
        try
        {
            if constexpr(std::is_same_v<std::decay_t<T>,Tensor>)
            {
                out.impl->val = (this->value() <= other.value());
            }
            else if constexpr(std::is_arithmetic_v<std::decay_t<T>>)
            {
                out.impl->val = (this->value() <= other);
            }
            else
            {
                static_assert(always_false<T>, "Unsupported type for Tensor comparison operator <=");
            }
            Logger::info("Successfully compared a tensor with a tensor or a number using <= operator");
        }
        catch(...)
        {
            Logger::error("Error while a comparing a tensor with a tensor or a number using <= operator");
            std::cerr << "Error while a comparing a tensor with a tensor or a number using <= operator" << std::endl;
        }
        return out;        
    }

    // > (greater than)
    template<typename T>
    Tensor operator>(T&& other)
    {
        Tensor out{};
        try
        {
            if constexpr(std::is_same_v<std::decay_t<T>,Tensor>)
            {
                out.impl->val = (this->value() > other.value());
            }
            else if constexpr(std::is_arithmetic_v<std::decay_t<T>>)
            {
                out.impl->val = (this->value() > other);
            }
            else
            {
                static_assert(always_false<T>, "Unsupported type for Tensor comparison operator >");
            }
            Logger::info("Successfully compared a tensor with a tensor or a number using > operator");
        }
        catch(...)
        {
            Logger::error("Error while a comparing a tensor with a tensor or a number using > operator");
            std::cerr << "Error while a comparing a tensor with a tensor or a number using > operator" << std::endl;
        }
        return out;        
    }

    // >= (greater than or equal)
    template<typename T>
    Tensor operator>=(T&& other)
    {
        Tensor out{};
        try
        {
            if constexpr(std::is_same_v<std::decay_t<T>,Tensor>)
            {
                out.impl->val = (this->value() >= other.value());
            }
            else if constexpr(std::is_arithmetic_v<std::decay_t<T>>)
            {
                out.impl->val = (this->value() >= other);
            }
            else
            {
                static_assert(always_false<T>, "Unsupported type for Tensor comparison operator >=");
            }
            Logger::info("Successfully compared a tensor with a tensor or a number using >= operator");
        }
        catch(...)
        {
            Logger::error("Error while a comparing a tensor with a tensor or a number using >= operator");
            std::cerr << "Error while a comparing a tensor with a tensor or a number using >= operator" << std::endl;
        }
        return out;        
    }

    Tensor pow(int num)
    {
        Tensor out{};
        double data_ = std::pow(this->value(),num);
        try
        {
            out.impl->val = data_;
            out.impl->prev = {this->impl};
            out.impl->backward_fn = [object = *this,num,out_impl = out.impl](){
                object.impl->grad += num * std::pow(object.value(),num-1) * out_impl->grad;
            };
            Logger::info("Successfully powered a tensor");
        }
        catch(...)
        {
            Logger::error("Error while powering a tensor");
            std::cerr << "Error while powering a tensor" << std::endl;
        }
        return out;
    }

    Tensor operator-() 
    {
        Tensor out{};
        try
        {
            out.impl->val = -1.0 * this->value();
            out.impl->prev = {this->impl};
            out.impl->backward_fn = [object = *this,&out](){
                object.impl->grad -= 1.0 * out.grad();
            };

            Logger::info("Successfully negated the tensor");
        }
        catch(...)
        {
            Logger::error("Error while negating the error");
            std::cerr << "Error while negating the error" << std::endl;
        }
        return out;
    }

    Tensor exp()
    {
        double data_ = this->impl->val;
        Tensor out{}; // keeping the out outside otherwise it will not be identified
        // by the file while compiling because of the scoping of the local variables
        try
        {
            out.impl->val = std::exp(data_);
            out.impl->prev = {this->impl};
            out.impl->backward_fn = [object = *this,out_impl = out.impl](){
                object.impl->grad += out_impl->val * out_impl->grad;
            };
            Logger::info("Successfully exponentiated a tensor");
        }
        catch(...)
        {
            Logger::error("Error in exponentiating");
            std::cerr << "Error in exponentiating" << std::endl;    
        }

        return out;
    }

    Tensor log()
    {
        double data_ = this->impl->val;
        Tensor out{};
        try
        {
            out.impl->val = std::log(data_);
            out.impl->prev = {this->impl};
            out.impl->backward_fn = [object = *this,data_,out_impl = out.impl](){
                object.impl->grad += 1/(data_) * out_impl->grad;
            };
            Logger::info("Successfully log a tensor");
        }
        catch(...)
        {
            Logger::error("Error while logarithmic");
            std::cerr << "Error while logarithmic" << std::endl;   
        }
        return out;
    }

    Tensor tanh()
    {
        double x = this->value();
        double t = (std::exp(2*x)-1)/(std::exp(2*x)+1);
        Tensor out{};
        try
        {
            out.impl->val = t;
            out.impl->prev = {this->impl};
            out.impl->backward_fn = [object = *this,t,out_impl = out.impl](){
                object.impl->grad += (1 - (t*t)) * out_impl->grad;
            };
            Logger::info("Successfully done the tanh function");
        }
        catch(...)
        {
            Logger::error("Error while doing with tanh");
            std::cerr << "Error while doing with tanh" << std::endl;
        }

        return out;
    }

    std::string shape()
    {
        return "()";
    }

    // friend std::ostream& operator<<(std::ostream& os,const Tensor& tensor)
    // {
    //     os << "Tensor(data=" << tensor.data << ")";
    //     return os;
    // }

    friend std::ostream& operator<<(std::ostream& os,const Tensor& tensor)
    {
        os << tensor.value();
        return os;
    }

    template<typename T1,class T2>
    typename std::enable_if<
        std::is_arithmetic_v<std::decay_t<T1>> &&
         std::is_same_v<std::decay_t<T2>,Tensor>,
        Tensor
    >::type
    friend operator+(T1&& number,T2&& tensor);

    template<typename T1,class T2>
    typename std::enable_if<
        std::is_arithmetic_v<std::decay_t<T1>> &&
         std::is_same_v<std::decay_t<T2>,Tensor>,
        Tensor
    >::type
    friend operator-(T1&& number,T2&& tensor); 

    template<typename T1,class T2>
    typename std::enable_if<
        std::is_arithmetic_v<std::decay_t<T1>> &&
         std::is_same_v<std::decay_t<T2>,Tensor>,
        Tensor
    >::type
    friend operator*(T1&& number,T2&& tensor);

    template<typename T1,class T2>
    typename std::enable_if<
        std::is_arithmetic_v<std::decay_t<T1>> &&
         std::is_same_v<std::decay_t<T2>,Tensor>,
        Tensor
    >::type
    friend operator/(T1&& number,T2&& tensor);

    template<typename T1,typename T2>
    typename std::enable_if<
        std::is_arithmetic_v<std::decay_t<T1>> &&
        std::is_same_v<std::decay_t<T2>,Tensor>,
        Tensor
    >::type
    friend operator==(T1&& number,T2&& tensor);

    template<typename T1,typename T2>
    typename std::enable_if<
        std::is_arithmetic_v<std::decay_t<T1>> &&
        std::is_same_v<std::decay_t<T2>,Tensor>,
        Tensor
    >::type
    friend operator!=(T1&& number,T2&& tensor);

    template<typename T1,typename T2>
    typename std::enable_if<
        std::is_arithmetic_v<std::decay_t<T1>> &&
        std::is_same_v<std::decay_t<T2>,Tensor>,
        Tensor
    >::type
    friend operator>(T1&& number,T2&& tensor);

    template<typename T1,typename T2>
    typename std::enable_if<
        std::is_arithmetic_v<std::decay_t<T1>> &&
        std::is_same_v<std::decay_t<T2>,Tensor>,
        Tensor
    >::type
    friend operator<(T1&& number,T2&& tensor);

    template<typename T1,typename T2>
    typename std::enable_if<
        std::is_arithmetic_v<std::decay_t<T1>> &&
        std::is_same_v<std::decay_t<T2>,Tensor>,
        Tensor
    >::type
    friend operator<=(T1&& number,T2&& tensor);

    template<typename T1,typename T2>
    typename std::enable_if<
        std::is_arithmetic_v<std::decay_t<T1>> &&
        std::is_same_v<std::decay_t<T2>,Tensor>,
        Tensor
    >::type
    friend operator>=(T1&& number,T2&& tensor);
};

// defining the templated methods or functions
// in the header file is recommemded
template<typename T1,class T2>
typename std::enable_if<
    std::is_arithmetic_v<std::decay_t<T1>> && std::is_same_v<std::decay_t<T2>,Tensor>,
    Tensor
>::type
operator+(T1&& number,T2&& tensor)
{
    Tensor out{};
    try
    {
        out.impl->val = number + tensor.value();
        out.impl->prev = {tensor.impl};
        // the out object might not exist later so 
        // it is good to use the shared pointer
        out.impl->backward_fn = [tensor_impl = tensor.impl,out_impl = out.impl](){
            tensor_impl->grad += out_impl->grad;
        };    

        Logger::info("Added a tensor to a number");
    }
    catch(...)
    {
        Logger::error("Error while adding a tensor to a number");
        std::cerr << "Error while adding a tensor to a number" << std::endl;
    }
    return out;
}

template<typename T1,class T2>
typename std::enable_if<
    std::is_arithmetic_v<std::decay_t<T1>> 
    && std::is_same_v<std::decay_t<T2>,Tensor>,
    Tensor
>::type
operator-(T1&& number,T2&& tensor)
{
        Tensor out{};
    try
    {
        out.impl->val = number - tensor.value();
        out.impl->prev = {tensor.impl};
        // the out object might not exist later so 
        // it is good to use the shared pointer
        out.impl->backward_fn = [tensor_impl = tensor.impl,out_impl = out.impl](){
            tensor_impl->grad -= out_impl->grad;
        };    

        Logger::info("subtracting a tensor to a number");
    }
    catch(...)
    {
        Logger::error("Error while subtracting a tensor from a number");
        std::cerr << "Error while substracting a tensor from a number" << std::endl;
    }
    return out;
}

template<typename T1,class T2>
typename std::enable_if<
    std::is_arithmetic_v<std::decay_t<T1>> 
    && std::is_same_v<std::decay_t<T2>,Tensor>,
    Tensor
>::type
operator*(T1&& number,T2&& tensor)
{
    Tensor out{};
    try
    {
        out.impl->val = number * tensor.value();
        out.impl->prev = {tensor.impl};
        // the out object might not exist later so 
        // it is good to use the shared pointer
        out.impl->backward_fn = [tensor_impl = tensor.impl,number,out_impl = out.impl](){
            tensor_impl->grad += number * out_impl->grad;
        };    

        Logger::info("subtracting a tensor to a number");
    }
    catch(...)
    {
        Logger::error("Error while subtracting a tensor from a number");
        std::cerr << "Error while substracting a tensor from a number" << std::endl;
    }
    return out;
}

template<typename T1,class T2>
typename std::enable_if<
    std::is_arithmetic_v<std::decay_t<T1>> 
    && std::is_same_v<std::decay_t<T2>,Tensor>,
    Tensor
>::type
operator/(T1&& number,T2&& tensor)
{
    Tensor out{};
    try
    {
        out.impl->val = number / tensor.value();
        out.impl->prev = {tensor.impl};
        // the out object might not exist later so 
        // it is good to use the shared pointer
        out.impl->backward_fn = [tensor_impl = tensor.impl,value = tensor.value(),out_impl = out.impl](){
            tensor_impl->grad += -1 * out_impl->grad / (std::exp(value,2));
        };    

        Logger::info("subtracting a tensor to a number");
    }
    catch(...)
    {
        Logger::error("Error while subtracting a tensor from a number");
        std::cerr << "Error while substracting a tensor from a number" << std::endl;
    }
    return out;
}

// in pytorch when we call backward function on boolean
// tensor , here it does not give an error
// it just does nothing

template<typename T1,typename T2>
typename std::enable_if<
    std::is_arithmetic_v<std::decay_t<T1>> &&
    std::is_same_v<std::decay_t<T2>,Tensor>,
    Tensor
>::type
operator==(T1&& number,T2&& tensor)
{
    Tensor out{};
    try
    {
        out.impl->val = (number == tensor.value());
        Logger::info("Successfully compared a number with a tensor using the == operator");
    }
    catch(...)
    {
        Logger::error("Error while comparing a number with a tensor using the == operator");
        std::cerr << "Error while comparing a number with a tensor using the == operator" << std::endl;
    }
    return out;
}

template<typename T1,typename T2>
typename std::enable_if<
    std::is_arithmetic_v<std::decay_t<T1>> &&
    std::is_same_v<std::decay_t<T2>,Tensor>,
    Tensor
>::type
operator!=(T1&& number,T2&& tensor)
{
    Tensor out{};
    try
    {
        out.impl->val = (number != tensor.value());
        Logger::info("Successfully compared a number with a tensor using the != operator");
    }
    catch(...)
    {
        Logger::error("Error while comparing a number with a tensor using the != operator");
        std::cerr << "Error while comparing a number with a tensor using the != operator" << std::endl;
    }
    return out;
}

template<typename T1,typename T2>
typename std::enable_if<
    std::is_arithmetic_v<std::decay_t<T1>> &&
    std::is_same_v<std::decay_t<T2>,Tensor>,
    Tensor
>::type
operator>(T1&& number,T2&& tensor)
{
    Tensor out{};

    try
    {
        out.impl->val = (number > tensor.value());
        Logger::info("Successfully compared a number with a tensor using the > operator");
    }
    catch(...)
    {
        Logger::error("Error while comparing a number with a tensor using the > operator");
        std::cerr << "Error while comparing a number with a tensor using the > operator" << std::endl;
    }
    return out;
}

template<typename T1,typename T2>
typename std::enable_if<
    std::is_arithmetic_v<std::decay_t<T1>> &&
    std::is_same_v<std::decay_t<T2>,Tensor>,
    Tensor
>::type
operator<(T1&& number,T2&& tensor)
{
    Tensor out{};

    try
    {
        out.impl->val = (number < tensor.value());
        Logger::info("Successfully compared a number with a tensor using the > operator");
    }
    catch(...)
    {
        Logger::error("Error while comparing a number with a tensor using the > operator");
        std::cerr << "Error while comparing a number with a tensor using the > operator" << std::endl;
    }
    return out;
}

template<typename T1,typename T2>
typename std::enable_if<
    std::is_arithmetic_v<std::decay_t<T1>> &&
    std::is_same_v<std::decay_t<T2>,Tensor>,
    Tensor
>::type
operator<=(T1&& number,T2&& tensor)
{
    Tensor out{};
    try
    {
        out.impl->val = (number <= tensor.value());
        Logger::info("Successfully compared a number with a tensor using the <= operator");
    }
    catch(...)
    {
        Logger::error("Error while comparing a number with a tensor using the <= operator");
        std::cerr << "Error while comparing a number with a tensor using the <= operator" << std::endl;
    }
    return out;
}

template<typename T1,typename T2>
typename std::enable_if<
    std::is_arithmetic_v<std::decay_t<T1>> &&
    std::is_same_v<std::decay_t<T2>,Tensor>,
    Tensor
>::type
operator>=(T1&& number,T2&& tensor)
{
    Tensor out{};

    try
    {
        out.impl->val = (number >= tensor.value());
        Logger::info("Successfully compared a number with a tensor using the >= operator");
    }
    catch(...)
    {
        Logger::error("Error while comparing a number with a tensor using the >= operator");
        std::cerr << "Error while comparing a number with a tensor using the >= operator" << std::endl;
    }
    return out;
}