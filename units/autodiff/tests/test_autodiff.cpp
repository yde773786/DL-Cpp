#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "computational_graph.hpp"

TEST_CASE( "Single Arithmetic Forward", "[arithmetic]" ) {
    ComputationalGraph graph;
    Node* a = new ImmutableNode(1);
    Node* b = new ImmutableNode(2);
    Node* c = new AddNode(0);
    graph.add_node(a);
    graph.add_node(b);
    graph.add_node(c);
    graph.add_connection(c, a);
    graph.add_connection(c, b);
    graph.forward();
    REQUIRE( c->value == 3 );
}