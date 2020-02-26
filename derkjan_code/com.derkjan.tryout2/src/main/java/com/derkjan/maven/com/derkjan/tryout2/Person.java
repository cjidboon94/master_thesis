package com.derkjan.maven.com.derkjan.tryout2;

import java.util.Set;
import java.util.HashSet;
import java.util.Random;

public class Person {
	enum State {
		SUSCEPTIBLE (0), INFECTED (1), RESISTANT (2);
		
		private int id;
		private State(int id) {
			this.id = id;
		}
		public int getId() {
			return id;
		}
	}
	public int id;
	private int diseaseTime;
	private int resistantTime;
	private int diseaseTimeLeft, resistantTimeLeft;
	public State state;
	public HashSet<Person> contacts;
	
	Person(int id, State state){
		this(id, state, 2, 2);
	}
	
	Person(int id, State state, HashSet<Person> contacts){
		this.id = id;
		this.state = state;
		this.contacts = contacts;
	}
	
	Person(int id, State state, int diseaseTime, int resistantTime){
		this.id = id;
		this.state = state;
		this.diseaseTime = diseaseTime;
		this.resistantTime = resistantTime;
	}
	
	Person(int id, State state, int diseaseTime, int resistantTime, HashSet<Person> contacts){
		this.id =id;
		this.state = state;
		this.diseaseTime = diseaseTime;
		this.resistantTime = resistantTime;
		this.contacts = contacts;
	}
	
	public void setContacts(int numberOfContacts) {
		contacts = new HashSet<Person>(numberOfContacts);	
	}
	
	public void addContact(Person contact) {
		contacts.add(contact);
	}
	
	public void setState(State newState) {
		state = newState;
		if (newState == State.INFECTED) {
			diseaseTimeLeft = diseaseTime;
		} else if (newState == State.RESISTANT) {
			resistantTimeLeft = resistantTime;
		}
	}
	
	public void updateState(double infectionChancePerContact) {
		Random rnd = new Random();
		if (state==State.RESISTANT && resistantTimeLeft >= 1) {
			resistantTimeLeft--;
		} else if (state==State.RESISTANT && resistantTimeLeft == 0) {
			resistantTimeLeft--;
			setState(State.SUSCEPTIBLE);
		} else if (state==State.INFECTED && diseaseTimeLeft >= 1) {
			diseaseTimeLeft--;
		} else if (state==State.INFECTED && diseaseTimeLeft == 0) {
			setState(State.RESISTANT);
		} else if (resistantTimeLeft == 0 || state == State.SUSCEPTIBLE) {
			if (rnd.nextFloat() < getInfectionChance(infectionChancePerContact)){
				setState(State.INFECTED);
			} else {
				setState(State.SUSCEPTIBLE);
			}			
		}
	}

	private double getInfectionChance(double infectionChancePerContact) {
		// TODO Auto-generated method stub
		double infectionChance = 1;
		for (Person contact : contacts) {
			if (contact.state == Person.State.INFECTED) {
				infectionChance *= infectionChancePerContact;
			}
		}

		return infectionChance;
	}
}
