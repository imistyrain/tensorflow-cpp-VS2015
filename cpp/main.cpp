#include <iostream>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
int testtensor()
{
    Scope root = Scope::NewRootScope();
    auto A = Const(root, { { 3.f, 2.f },{ -1.f, 0.f } });
    auto b = Const(root, { { 3.f, 5.f } });
    auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
    std::vector<Tensor> outputs;
    ClientSession session(root);
    TF_CHECK_OK(session.Run({ v }, &outputs));
    std::cout << outputs[0].matrix<float>();
    return 0;
}

int testloadgraph()
{
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        cout << status.ToString() << "\n";
        return 1;
    }
    cout << "Session successfully created.\n";
    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), "../models/add_graph.pb", &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }
    else {
        std::cout << "Load graph protobuf successfully" << std::endl;
    }
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }
    else {
        std::cout << "Add graph to session successfully" << std::endl;
    }
    Tensor a(DT_FLOAT, TensorShape());
    a.scalar<float>()() = 3.0;
    Tensor b(DT_FLOAT, TensorShape());
    b.scalar<float>()() = 2.0;
    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
        { "a", a },
        { "b", b },
    };
    std::vector<tensorflow::Tensor> outputs;
    status = session->Run(inputs, { "c" }, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }
    else {
        std::cout << "Run session successfully" << std::endl;
    }
    auto c = outputs[0].scalar<float>();
    std::cout << "output value: " << c() << std::endl;
    session->Close();
    return 0;
}

int main()
{
    testtensor();
    testloadgraph();
    return 0;
}