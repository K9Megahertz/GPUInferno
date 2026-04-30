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
		
		std::vector<std::pair<std::string, Tensor*>> parameters() const;

		void register_parameter(const std::string& name, Tensor* tensor);
		void register_module(const std::string& name, Module* module);
		void register_buffer(const std::string& name, Tensor* tensor);		
		StateDict state_dict() const;		
		void load_state_dict(const StateDict& state);
		void to(const Device& device);
		void collect_state_dict(StateDict& out, const std::string& prefix = "") const;
		bool check_name_exists(std::string name);

		void collect_parameters(std::vector<std::pair<std::string, Tensor*>>& out, const std::string& prefix) const;

		void load_state_dict_recursive(const StateDict& state, const std::string& prefix);
	

	private:

		
	
		
		std::vector<std::pair<std::string, Module*>> _children;		
		std::vector<std::pair<std::string, Tensor*>> _parameters;		
		std::vector<std::pair<std::string, Tensor*>> _buffers;

	};

	std::ostream& operator<<(std::ostream& os, const Module& module);


}