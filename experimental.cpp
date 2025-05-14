#include <memory>
#include <functional>
#include <iostream>
#include <vector>
#include <string>

class Human
{
    public:
        std::function<void()> _function;
        std::vector<std::shared_ptr<Human>> children;
        std::string name;
        static int count;

        Human() : name(std::to_string(count)) {count++;};
};

int Human::count = 0;
Human operator+(Human& human1,Human& human2)
{
    Human out{};
    out.children = human1.children;
    out.children.insert(out.children.end(),human2.children.begin(),human2.children.end());
    out._function = [human_one = human1,&out,human_two = human2](){
        if(human_one._function) human_one._function();
        if(human_two._function) human_two._function();
        std::cout << "Hi my name is" << out.name << std::endl;
    };
    return out;
}

// Human& operator+=(Human& human1,Human&& human2)
// {
//     auto prev_function = human1._function;
//     human1.children.insert(human1.children.end(),human2.children.begin(),human2.children.end());
//     human1._function = [prev_function,](){

//     };
//     return human1;
// }

class Object
{
    public:
    std::vector<std::reference_wrapper<Object>> child;
    std::vector<std::shared_ptr<Object>> ownership;
    std::function<void()> _function;
    std::string name;
    Object(std::string name) : name(name),child({}),ownership({}),_function([](){}) {}
    Object(std::string name,Object& child) : name(name),child({std::ref(child)}),ownership({}),_function([](){}) {}

    friend std::ostream& operator<<(std::ostream& os,const Object& object)
    {
        os << object.name << std::endl;
        for(int i = 0;i < object.child.size();i++)
        {
            os << object.child[i] << std::endl;
        }
        return os;
    }

    bool operator==(Object& object) const
    {
        return this == &object;
    }
};

Object& function2(Object& obj1,Object&& obj2)
{
    obj1.child.insert(obj1.child.end(),obj2.child.begin(),obj2.child.end());
    return obj1;
}

Object& function3(Object& obj1,Object&& obj2)
{
    // the without std::move performs a copy of the rvalue
    std::shared_ptr<Object> obj2_ptr = std::make_shared<Object>(obj2);
    obj1.child.push_back(std::ref(*obj2_ptr));
    // i comment the line below so no one has ownership
    // of this shared pointer so it destroyed
    obj1.ownership.push_back(obj2_ptr); // after taking the ownership 
    // it works fine
    return obj1;
}

// ERROR FUNCTION
// CAUTION!
Object& function4(Object& obj1,Object&& obj2)
{
    obj1._function = [&obj2](){
        std::cout << obj2 << std::endl;
        // Pretty sure should give segmentation core dump
    };
    return obj1;
}

Object& function5(Object& obj1,Object&& obj2)
{
    // with std::move making it more optimized by moving the ownership
    // to someone else and give the resources that entity
    // and does not destroyed content in that memory 
    // that content just remains there untouched
    std::shared_ptr<Object> obj2_ptr = std::make_shared<Object>(std::move(obj2));
    obj1.child.push_back(std::ref(*obj2_ptr));
    // since the lambda function has taken the ownership of the shared_ptr
    // object so the object is not destroyed
    obj1._function = [obj2_ptr](){
        std::cout << *obj2_ptr << std::endl;
    };
    return obj1;
}

int main()
{
    Object child{"Anubhav"};
    Object obj1 = {"Anirban"};
    obj1 = function5(obj1,Object{"Jayashree",child});
    // since the outcome results in undefined behaviour
    // so the object rvalue is destroyed 
    // Yes, the fact that std::string uses heap memory 
    // can make undefined behavior appear to "work" — but 
    // it’s not reliable or safe. The proper fix is to ensure
    //  the object outlives the reference — by naming it or using shared_ptr.
    // std::cout << obj1 << std::endl;
    // Jayashree is getting printed since jayashree string is created
    // in the heap but if the object is destroyed the thing
    // is it cannot be reclaimed back and it is like
    // smart pointers , they get removes automatically from the heap
    obj1._function();
    std::cout << obj1 << std::endl;
    // ->get is used to get the object and the reference of the object is compared
    // function2 inserts the object by reference
    // if (&(obj1.child.begin()->get()) == &child)
    // {
    //     std::cout << "They point to the same element" << std::endl;
    // }
    // else
    // {
    //     std::cout << "They do not point to the same element" << std::endl;
    // }
}
