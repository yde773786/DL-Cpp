#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "../computational_graph.hpp"
#include "../operations.hpp"
#include "../loss_fns.hpp"
#include "../activations.hpp"

TEST_CASE( "Single Arithmetic", "[arithmetic]" ) {
    ComputationalGraph graph;
    Node* a = new ChildlessNode(1);
    Node* b = new ChildlessNode(2);
    Node* c = new AddNode(0);
    graph.add_node(a);
    graph.add_node(b);
    graph.add_node(c);
    graph.add_connection(c, a);
    graph.add_connection(c, b);

    graph.forward();
    REQUIRE( c->value == 3 );
    REQUIRE( a->value == 1 );
    REQUIRE( b->value == 2 );

    c->gradient = 2;
    graph.backward();
    REQUIRE( a->gradient == 1 );
    REQUIRE( b->gradient == 1 );
}

TEST_CASE( "Perceptron", "[dl]" ){
    ComputationalGraph graph;
    Node* x1 = new ChildlessNode(1);
    Node* x2 = new ChildlessNode(1);

    Node* w1 = new ChildlessNode(0.5);
    Node* w2 = new ChildlessNode(0.5);
    Node* b = new ChildlessNode(0);

    Node* w1_x1 = new MulNode(0);
    Node* w2_x2 = new MulNode(0);
    Node* w1_x1_w2_x2_b = new AddNode(0);

    Node* y_bar = new TanhNode(0);
    Node* y = new ChildlessNode(1);
    Node* loss = new MSENode(0);

    graph.add_node(x1);
    graph.add_node(x2);
    graph.add_node(w1);
    graph.add_node(w2);
    graph.add_node(b);
    graph.add_node(w1_x1);
    graph.add_node(w2_x2);
    graph.add_node(w1_x1_w2_x2_b);
    graph.add_node(y_bar);
    graph.add_node(y);
    graph.add_node(loss);

    graph.add_connection(w1_x1, w1);
    graph.add_connection(w1_x1, x1);
    graph.add_connection(w2_x2, w2);
    graph.add_connection(w2_x2, x2);

    graph.add_connection(w1_x1_w2_x2_b, w1_x1);
    graph.add_connection(w1_x1_w2_x2_b, w2_x2);
    graph.add_connection(w1_x1_w2_x2_b, b);

    graph.add_connection(y_bar, w1_x1_w2_x2_b);
    graph.add_connection(loss, y_bar);
    graph.add_connection(loss, y);
    
    ((MSENode*) loss)->add_output_target_pair(y_bar, y);

    graph.forward();
    graph.backward();
}