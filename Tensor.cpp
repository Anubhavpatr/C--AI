#include "Tensor.h"
#include <unordered_set>

Tensor::Tensor()
    : impl(std::make_shared<Impl>(0.0f)) {}


Tensor::Tensor(float val) : impl(std::make_shared<Impl>(val)) {}

float Tensor::value() const {
    return impl->val;
}



float Tensor::grad() const {
    return impl->grad;
}

void Tensor::backward() {
    impl->grad = 1.0f;

    std::vector<std::shared_ptr<Impl>> topo_order;
    std::unordered_set<Impl*> visited;

    std::function<void(std::shared_ptr<Impl>)> dfs;
    dfs = [&](std::shared_ptr<Impl> node) {
        if (!node || visited.count(node.get())) return;
        visited.insert(node.get()); // used to get the raw pointer
        for (auto& prev : node->prev) {
            dfs(prev);
        }
        topo_order.push_back(node);
    };

    dfs(impl);

    // reverse topological order to propagate gradients
    std::reverse(topo_order.begin(), topo_order.end());
    for (auto& node : topo_order) {
        if (node->backward_fn) node->backward_fn();
    }
}


// Use pass-by-value to support lvalues and rvalues equally
// Tensor operator+(Tensor lhs, Tensor rhs) {
//     // lhs and rhs are copies 
//     // when we copy shared_ptr it points to 
//     // the same memory it's fine
//     Tensor out(lhs.value() + rhs.value());
//     // i am getting the pointers of the function,
//     // value , grad and it's prev
//     // since if pointers are copied
//     // they copy the reference
//     // so they point to the same location
//     out.impl->prev = {lhs.impl, rhs.impl};
//     out.impl->backward_fn = [lhs_impl = lhs.impl, rhs_impl = rhs.impl, out_impl = out.impl]() {
//         lhs_impl->grad += out_impl->grad;
//         rhs_impl->grad += out_impl->grad;
//     };
//     return out;
// }

// Tensor operator*(Tensor lhs, Tensor rhs) {
//     Tensor out(lhs.value() * rhs.value());
//     out.impl->prev = {lhs.impl, rhs.impl};
//     // as long as you make the shared_pointer
//     // owned by someone if it is the shared pointer 
//     // of an lvalue or rvalue
//     // that object which the pointer is pointing is still there

//     // just ensure that the shared_ptr which points to the actual
//     // content is owned by someone like
//     //1) a variable
//     //2) a lambda function
//     //3) a container or data structure

//     //  std::move is used to transfer ownership
//     //  std::forward is used to forward the object
//     // with it's type whether it's an rvalue or lvalue

//     // So even if:

//     // The original object (e.g., a temporary or rvalue like Tensor{...}) goes out of scope,

//     // If a shared_ptr was created from it and still owned by some other thing (somewhere), the object remains alive.



//     // since the parameters persist impl pointers point to the correct memory location
//     out.impl->backward_fn = [lhs_impl = lhs.impl, rhs_impl = rhs.impl, out_impl = out.impl]() {
//         lhs_impl->grad += rhs_impl->val * out_impl->grad;
//         rhs_impl->grad += lhs_impl->val * out_impl->grad;
//     };
//     return out;
// }

// Tensor& Tensor::operator+=(const Tensor& rhs) {
//     *this = *this + rhs;  // Leverage autograd-tracked +
//     // since the parameters persist impl pointers point to the correct memory location
//     return *this;
// }

// Tensor& Tensor::operator+=(Tensor&& rhs) {
//     *this = *this + std::move(rhs);  // Rvalue-aware version
//     // transfering the ownership of the content of rhs to *this object
//     // because rhs will not be able to survive much
//     // destroying the rvalue and moving inside the this
//     // std::move avoids deep copy so theyfore the
//     // attributes of the copy in the function
//     // point the same memory location
//     return *this;
// }