#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;
/// Edit this file to define custom logic or remove it if it is not needed.
/// Learn more about FRAME and the core library of Substrate FRAME pallets:
/// https://substrate.dev/docs/en/knowledgebase/runtime/frame

use frame_support::{decl_module, decl_storage, decl_event,ensure, decl_error, dispatch, traits::Get};
use frame_system::ensure_signed;
use frame_support::codec::{Encode, Decode};

use engine::nl::NeuralLayer;
use engine::nn::NeuralNetwork;
use engine::activation::{
	HyperbolicTangent,
	Sigmoid,
	Identity,
	LeakyRectifiedLinearUnit,
	RectifiedLinearUnit,
	SoftMax,
	SoftPlus	
};
use engine::activation::{
	ActivationArgu,
	ActivationKind,
	Activation
};

use engine::cost::{
	CostFunction,
	CostFunctions,
	CostFunctionArgu
};


use engine::sample::Sample;
use engine::matrix::Matrix;

use core::str;

pub mod engine;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

use crate::alloc::vec::Vec;

pub trait Trait: frame_system::Trait {
	type Event: From<Event<Self>> + Into<<Self as frame_system::Trait>::Event>;
}

type NeuralKey<AcId> = (AcId,Vec<u8>); 

#[derive(Encode, Decode,Default, Clone, PartialEq,Hash)]
pub struct NeuralStruct {
	name: Vec<u8>,
	neural_network: Option<Vec<u8>>
}

impl NeuralStruct{
	pub fn new(name: Vec<u8>)-> Self{
		NeuralStruct{
			name: name,
			neural_network: None
		}
	}
	pub fn get_model(&self) -> Result<NeuralNetwork,Vec<u8>>{
		if let Some(nn)=self.neural_network.clone(){
			NeuralNetwork::get_neural_from_str(nn)
		}else{
			Err("No Model".as_bytes().to_vec())
		}
	} 
	pub fn add_layers(&mut self,layer: NeuralLayer){
		if self.neural_network.is_none(){
			self.neural_network=Some(NeuralNetwork::new().get_serial());
		}
		if let Ok(mut net) = self.get_model(){
			net.add_layer(layer);
			self.neural_network=Some(net.get_serial());
		}	
	}
	pub fn get_model_string(&self)->Vec<u8>{
		self.neural_network.clone().unwrap_or(vec![0])
	}
	pub fn train(&mut self,samples: Vec<Sample> ,epoch: i32, learning_rate: f64){
		let mut nn =self.get_model().unwrap();
		nn.train(samples,epoch,learning_rate,None);
		self.neural_network=Some(nn.get_serial());
	}
	pub fn run(&self, samples: Sample) -> Matrix{
		let nn =self.get_model().unwrap();
		nn.evaluate(&samples)
	}
}

decl_storage! {
	trait Store for Module<T: Trait> as TemplateModule{
		NeuralContainer get(fn neural_container): map hasher(blake2_128_concat) (T::AccountId,Vec<u8>) => NeuralStruct;
		DataContainer get(fn data_container): map hasher(blake2_128_concat) (T::AccountId,Vec<u8>) => Vec<Vec<u8>>;
	}
}

decl_event!(
	pub enum Event<T> where AccountId = <T as frame_system::Trait>::AccountId {
		MakeNewModel(NeuralKey<AccountId>),
		AddLayer(NeuralKey<AccountId>,Vec<u8>),
		UpdateModel(NeuralKey<AccountId>,Vec<u8>),
		AddDataSet(NeuralKey<AccountId>),
		TrainComplete(NeuralKey<AccountId>),
		RunResult(NeuralKey<AccountId>,Vec<u8>),
	}
);

// Errors inform users that something went wrong.
decl_error! {
	pub enum Error for Module<T: Trait> {
		UnableNewNueral,
		NoModel,
		WrongLayerType,
		ModelParsingError,
		NoData
	}
}

fn read_input_data(data: Vec<u8>) ->Vec<Vec<f64>> {
	let s = str::from_utf8(data.as_slice()).unwrap();
	let lines = s.split('\n');
	let mut result = Vec::new(); 
	for l in lines{
		let d = l.split(',').map(|s| s.parse().unwrap()).collect();
		result.push(d);
	}
	result
}

decl_module! {
	pub struct Module<T: Trait> for enum Call where origin: T::Origin {
		// Errors must be initialized if they are used by the pallet.
		type Error = Error<T>;

		// Events must be initialized if they are used by the pallet.
		fn deposit_event() = default;
		
		// generate new model
		#[weight = 1_100_000_000]
		pub fn make_new_neural(origin, name: Vec<u8>) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;
			
			let ns = NeuralStruct::new(name.clone()); 
			Self::deposit_event(RawEvent::MakeNewModel((who.clone(),name.clone())));
			<NeuralContainer<T>>::insert((who.clone(),name.clone()), ns);
			Ok(())
		}
		
		//add layer to model
		#[weight = 2_500_000_000]
		pub fn add_layer(origin,name: Vec<u8>,size: (u32,u32),layer_type: Vec<u8>, extra_parameter: u64) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;
			let (in_s,out_s) = (size.0 as usize, size.1 as usize);

			ensure!(<NeuralContainer<T>>::contains_key(&(who.clone(),name.clone())),Error::<T>::NoModel);

			let mut ns=<NeuralContainer<T>>::get((who.clone(),name.clone()));


			let layer = match str::from_utf8(layer_type.as_slice()).unwrap(){
				"HyperBolicTangent" =>{
					Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::HyperbolicTangent,vec![0f64])))
				},
				"Sigmoid" =>{
					Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::Sigmoid,vec![0f64])))
				},
				"RectifiedLinear" =>{
					Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::RectifiedLinearUnit,vec![0f64])))
				}
				"LeackyLelu" =>{
					Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::LeakyRectifiedLinearUnit,vec![extra_parameter as f64])))
				},
				"SoftMax" =>{
					Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::SoftMax,vec![0f64])))
				},
				"SoftPlus" =>{
					Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::SoftPlus,vec![0f64])))
				},
				"Identity" =>{
					Some(NeuralLayer::new(out_s,in_s,ActivationArgu::new(ActivationKind::Identity,vec![0f64])))
				},
				_ =>{
					None
				}
			};
			ensure!(layer.is_some(),Error::<T>::WrongLayerType);
			ns.add_layers(layer.unwrap());
			Self::deposit_event(RawEvent::UpdateModel((who.clone(),name.clone()),ns.get_model_string()));
			<NeuralContainer<T>>::mutate((who.clone(),name.clone()),move |i|{
				*i=ns;
			});
			Ok(())
		}

		//add data_set for model
		#[weight = 2_500_000_000]
		pub fn add_data_set(origin,name: Vec<u8>,size: (u32,u32), input_data: Vec<u8> ) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;

			ensure!(<NeuralContainer<T>>::contains_key(&(who.clone(),name.clone())),Error::<T>::NoModel);

			let (in_s,out_s) = (size.0 as usize, size.1 as usize);

			let mut samples = Vec::new();
			let datas = read_input_data(input_data);
			for l in datas {
				let result:Vec<f64> = l.iter().skip(in_s).take(out_s).map(|i| *i).collect();
				let input: Vec<f64> = l.iter().take(in_s).map(|i| *i).collect();
				let sample = Sample::new(input,result);
				samples.push(sample.get_serial());
			}

			<DataContainer<T>>::insert((who.clone(),name.clone()), samples);
			
			Self::deposit_event(RawEvent::AddDataSet((who.clone(),name.clone())));
			Ok(())
		}

		//train model
		#[weight = 2_500_000_000]
		pub fn train(origin,name: Vec<u8>,epoch: i32, learning_rate: u32) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;

			ensure!(<NeuralContainer<T>>::contains_key(&(who.clone(),name.clone())),Error::<T>::NoModel);

			let mut ns=<NeuralContainer<T>>::get((who.clone(),name.clone()));
			ensure!(ns.get_model().is_ok(),Error::<T>::ModelParsingError);

			ensure!(<DataContainer<T>>::contains_key(&(who.clone(),name.clone())),Error::<T>::NoData);
			let datas = <DataContainer<T>>::get((who.clone(),name.clone())); 

			let mut samples = Vec::new();
			for raw_string in datas.iter(){
				match Sample::from_string(raw_string.clone()){
					Ok(s) => {
						samples.push(s)
					},
					Err(_) =>{
						
					}
				}
			}
			ns.train(samples,epoch,learning_rate as f64);
			Self::deposit_event(RawEvent::TrainComplete((who.clone(),name.clone())));
			<NeuralContainer<T>>::mutate((who.clone(),name.clone()),move |i|{
				*i=ns
			});
			Ok(())
		}

		//run model
		#[weight = 2_500_000_000]
		pub fn run(origin,name: Vec<u8>, input_data: Vec<u8>) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;

			ensure!(<NeuralContainer<T>>::contains_key(&(who.clone(),name.clone())),Error::<T>::NoModel);
			let ns=<NeuralContainer<T>>::get((who.clone(),name.clone()));

			ensure!(ns.get_model().is_ok(),Error::<T>::ModelParsingError);
			let sample = read_input_data(input_data);
			let sample = sample[0].clone();
			let result=ns.run(Sample::predict(sample));
			
			Self::deposit_event(RawEvent::RunResult((who.clone(),name.clone()),result.get_serial()));
			Ok(())
		}
	}
}