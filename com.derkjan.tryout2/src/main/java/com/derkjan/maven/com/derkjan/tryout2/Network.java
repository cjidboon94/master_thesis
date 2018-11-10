package com.derkjan.maven.com.derkjan.tryout2;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

public class Network {
	HashMap<Integer, Person> idToPersons;
	HashMap<Integer, Integer> idToIndex;
	int[] states;
	
	public Network(int numberOfPersons) {
		idToPersons = new HashMap<Integer, Person>(numberOfPersons); 
	}
	
	public Network(HashMap<Integer, Person> idToPersons) {
		this.idToPersons = idToPersons;
	}
	
	public void addPerson(int id, Person person) {
		idToPersons.put(id, person);
	}
	
	public Person getPerson(int id) {
		return idToPersons.get(id);
	}

	public Object getPersons() {
		return idToPersons.values();
	}
	
	public void setIdToIndex() {
		idToIndex = getIdToIndex(idToPersons.values().size());
	}
	
	private HashMap<Integer, Integer> getIdToIndex(int numberOfPersons) {
		HashMap<Integer, Integer> idToIndex = new HashMap<Integer, Integer>(numberOfPersons);
		Set<Integer> ids = idToPersons.keySet();
		List<Integer> list = new ArrayList<Integer>(ids);
		Collections.sort(list);
		int count = 0;
		for (int id: list) {
			idToIndex.put(id, count);
			count++;
		}
		return idToIndex;
	}
	
	/**
	 * function to update all persons in network ones
	 */
	public void updateNetwork(double infectionChance) {
		Set<Integer> ids = idToPersons.keySet();
		List<Integer> list = new ArrayList<Integer>(ids);
		Collections.shuffle(list);
		for (int id : list) {
			idToPersons.get(id).updateState(infectionChance);
		} 
		//states = getState();
	}
	
	/**
	 * 
	 * @return
	 */
	public int[] getState() {
		Collection<Person> persons = (Collection<Person>) idToPersons.values(); 
		int[] states = new int[persons.size()];
		for (Person person: persons) {
			states[idToIndex.get(person.id)] = person.state.getId(); 
		}
		return states;
	}
	
	public HashMap<Integer, Person> getIdToPersons() {
		return idToPersons;
	}
}
