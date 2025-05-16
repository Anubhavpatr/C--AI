#include "Tensor.h"
#include "Matrix.h"
// // #include "DataLoader.h"
// // #include <set>
// // #include <map>
// // #include <random>
// // #include <algorithm>

int main()
{
    auto ptr1 = std::shared_ptr<Tensor[]>(new Tensor[6]{1,2,3,4,5,6});
    Matrix<Tensor> m{2,3,ptr1}; // first matrix
    auto ptr2 = std::shared_ptr<Tensor[]>(new Tensor[6]{1,2,3,4,5,6});
    Matrix<Tensor> m2{3,2,ptr2}; // second matrix


    // Matrix<Tensor> m10 = Matrix<Tensor>::ones_like(m2);
    // m10.print();

    // Matrix<Tensor> m11 = Matrix<Tensor>::zeros_like(m2);
    // m11.print();

    // Matrix<Tensor> m12 = m2.sum(0,false);
    // Tensor t = m12.sum();
    // m2.print();
    // m12.print();
    // t.backward();
    // std::cout << t << std::endl;
    // std::cout << m2[{1,0}].grad() << std::endl;
    // std::cout << m2[{2,1}].grad() << std::endl;

    // Matrix<Tensor> m_clone = m2.clone();
    // m_clone.print();


    // auto ptr3 =  std::shared_ptr<Tensor[]>(new Tensor[2]{1,2});
    // Matrix<Tensor> m3{2,1,ptr3};

    // Matrix<Tensor> m6 = m2.sum(1, true); // result is OneOrTwoD
    // m6.print();
    // std::cout << m6.shape() << std::endl;
    // // the sum works and the gradients works
    // m6[{2,0}].backward();
    // std::cout << m2[{2,1}].grad() << std::endl;
    // std::cout << m6.shape(0) << std::endl;

    // Matrix<Tensor> m7 = m2.pow(2);
    // m7.print();
    // m2.print();

    // m7[{1,1}].backward();
    // std::cout << m2[{1,1}].grad() << std::endl;

    // Matrix<Tensor> m8 = m2.max(1,true);
    // m8.print();
    // m8[{2,0}].backward();
    // std::cout << m2[{2,1}].grad() << std::endl;

    // Matrix<Tensor> m9 = m2.min(1,true);
    // m9.print();
    // m9[{2,0}].backward();
    // std::cout << m2[{2,0}].grad() << std::endl;

    //Matrix<Tensor> m4 = m.matmul(m2); // matrix multiplication
    // m4 shape (2,2) - m3 shape (2,1)
    //Matrix<Tensor> m5 = m4 * m3; 
    // m4.print();
    // m3.print();
    //m5.print();
    // the broadcasting rules work
    //m5[{1,1}].backward();
    //std::cout << m3[{1,0}].grad() << std::endl;
    // std::cout << m3[{1,1}] << std::endl;
    //m4.print();
    // Matrix<Tensor> m3_T = m3.transpose();
    // m3_T.print();
    // m3_T[{1,1}].backward();
    // std::cout << m[{1,1}].grad() << std::endl; // gradients are working fine with matrices
}


// int main()
// {
//     std::vector<std::string> words = read_file("names.txt");
//     std::string joined;
//     for(const auto& word : words)
//     {
//         joined += word;
//     }

//     std::set<char> letters(joined.begin(),joined.end());
//     std::vector<char> chars(letters.begin(),letters.end());
//     std::map<char,int> ctoi{};
//     for(int i = 0;i < chars.size();i++)
//     {
//         ctoi[chars[i]] = i+1;
//     }
//     ctoi['.'] = 0;
//     std::map<int,char> itoc{};
//     for(const auto& [key,value] : ctoi)
//     {
//         itoc[value] = key;
//     }

//     int vocab_size = ctoi.size();
//     int block_size = 3; // context length: how many characters do we take to predict the next one?
//     std::tuple<std::vector<std::vector<int>>,std::vector<int>> get_x_y = 
//     build_dataset(words,block_size,ctoi);
//     std::vector<std::vector<int>> X = std::get<0>(get_x_y);
//     std::vector<int> Y = std::get<1>(get_x_y);
//     std::cout << "Reached Here" << std::endl;
    
//     // for(int i = 0;i < X.size();i++)
//     // {
//     //     for(int j = 0;j < X[i].size();j++)
//     //     {
//     //         std::cout << X[i][j] << " ";
//     //     }
//     //     std::cout << std::endl;
//     // }

//     std::mt19937 rng(42); // fixed seed for reproducibility
//     std::shuffle(words.begin(), words.end(), rng);

//     int n1 = static_cast<int>(0.8 * words.size());
//     int n2 = static_cast<int>(0.9 * words.size());

//     std::vector<std::string> words_tr(words.begin(), words.begin() + n1);
//     std::vector<std::string> words_dev(words.begin() + n1, words.begin() + n2);
//     std::vector<std::string> words_te(words.begin() + n2, words.end());

//     auto [Xtr, Ytr] = build_dataset(words_tr, block_size, ctoi);
//     auto [Xdev, Ydev] = build_dataset(words_dev, block_size, ctoi);
//     auto [Xte, Yte] = build_dataset(words_te, block_size, ctoi);
//     int embedding_dim = 8;
//     Matrix<Tensor> C{vocab_size,embedding_dim,0.02};
//     ThreeDArray<Tensor> Embeddings_out = C[Xtr];
//     // Embeddings_out.print();

//     std::vector<std::shared_ptr<Tensor>> out = Embeddings_out[{2,3}];
//     for(int i = 0;i < out.size();i++)
//         std::cout << *out[i] << " ";
//     std::cout << std::endl;


//     // std::cout << "X size" << X.size() << std::endl;
// }






/////// TESTING FOR AUTOGRAD ENGINE ////////////////////////////////

// #include "Tensor.h"
// #include <iostream>

// int main() {
//     Logger::basicConfig("Logger.txt",Logger::Loggermode::OPTIMIZED);
//     Tensor a = 1.0f;
//     Tensor b = 2.0f;
//     // Tensor c = 4.0f;
//     // Tensor e = 5.0f;
//     //std::cout << "Hello" << std::endl;
//     //std::cout << "hello" << std::endl;
//     Tensor d{5};
//     d *= 2 * b;
//     // do not assign it toitself it will overwrite the tensor
//     d = d.log();
//     d = d.pow(3);
//     Tensor e = d;
//     Tensor f = e.sigmoid();
//     f.backward();

//     std::cout << "e: " << e.value() << std::endl;
//     std::cout << "d: " << d.value() << std::endl;
//     std::cout << "d.grad: " << d.grad() << std::endl;
//     std::cout << "a.grad: " << a.grad() << std::endl; // should be 3.0
//     std::cout << "b.grad: " << b.grad() << std::endl; // should be 2.0
//     // std::cout << "c.grad: " << c.grad() << std::endl; // should be 5.0
//     // std::cout << "e.grad: " << e.grad() << std::endl; // should be 4.0
// }
