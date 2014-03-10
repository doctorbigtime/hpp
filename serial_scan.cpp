#include <iterator>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

struct Adder
{
   template < typename T >
   T operator()( T a, T b ) const
   { return a+b; }
};

template < typename T >
std::vector<T> importVector( const std::string& filename )
{
   std::ifstream ifs( filename.c_str() );
   if( !ifs ) return std::vector<T>();
   std::istream_iterator<T> it( ifs );
   std::vector< T > v;
   for( ; std::istream_iterator<T>() != it; ++it )
      v.push_back( *it );
   return v;
}

int main(int argc, char ** argv) 
{
   int numElements; // number of elements in the list

   std::vector<uint64_t> hostOutput;
   std::vector<uint64_t> hostInput(importVector<uint64_t>(argv[1]));
   numElements = (int)hostInput.size();

   std::cout << "numElements: " << numElements << std::endl;

   hostOutput.resize( numElements );

   hostOutput[0] = hostInput[0];
   for( int i = 1; i < numElements; ++i )
   {
      hostOutput[i] = Adder()( hostOutput[i-1], hostInput[i] );
      //hostOutput[i] = hostOutput[i-1] + hostInput[i];
   }

   if( numElements < 100 )
      std::copy( hostOutput.begin(), hostOutput.end()
      , std::ostream_iterator<uint64_t>(std::cout,"\n") );
   else
      std::cout << "lastElement: " << hostOutput[numElements-1] << "\n";

   return 0;
}

