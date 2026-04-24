#pragma once
#include <inferno/core/tensor.h>
#include <inferno/core/ops.h>
#include <unordered_map>
#include <map>






namespace Inferno {



	using StateDict = std::map<std::string, Tensor>;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Class Module
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class Module {

	public:

		virtual ~Module() = default;
		virtual Tensor forward(Tensor& input);

		std::vector<Tensor*> parameters() const;

		void register_parameter(const std::string& name, Tensor* tensor);
		void register_module(const std::string& name, Module* module);
		void register_buffer(const std::string& name, Tensor* tensor);		
		StateDict state_dict() const;		
		void to(const Device& device);
		void collect_state_dict(StateDict& out, const std::string& prefix = "") const;

	

	private:

		
	
		std::unordered_map<std::string, Module*> _children;
		std::unordered_map<std::string, Tensor*> _parameters;
		std::unordered_map<std::string, Tensor*> _buffers;

	};

	std::ostream& operator<<(std::ostream& os, const Module& module);


}