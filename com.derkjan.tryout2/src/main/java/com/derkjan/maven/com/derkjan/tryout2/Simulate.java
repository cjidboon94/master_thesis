package com.derkjan.maven.com.derkjan.tryout2;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.Collection;
import java.util.Random;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonReader;

/*
 * This class is meant to start a simulation of a disease network and return
 * a number of samples indicating the network state at certain times
 * 
 * This package should be run as follows: 
 * 
 * java Simulate networkFile percentageInitialInfected InfectionChance numberOfSamples
 *
 * The networkFile should have the following format
 * 
 * It returns the filename where the 
 */


public class Simulate {
	static double initialInfectedPercentage = 0.05;
	
	public static void main(String[] args) throws FileNotFoundException {
		//String inputFile = args[0];
		//double initialInfectedPercentage = Double.parseDouble(args[1]);
		//double InfectionChance = Double.parseDouble(args[2]);
		//int numberOfSamples = Integer.parseInt(args[3]);
		
		//for (String arg : args) {
		//	System.out.println(arg);
		//}
		//Person person1 = new Person(Person.State.SUSCEPTIBLE);

		// settings
	    int numberOfNetworks = 100;
	    int numberOfNodes = 100;
	    int networkDegree = 2;
	    String mode = "powerlaw";
	    String filePath = "/home/derkjan/Documents/academics_UVA/master_thesis_code/master_thesis/";
	    String directory = "network_files/";
	    String fileFormat = "network%dnodes%ddegree%dmode_%s.json";
	    int numberOfSimulations = 100000;
	    int burnInPeriod = 1000;
	    double infectionChance = 0.1;
	    String directoryOut = "networkDiseaseFiles/";
	    String fileFormatOut = "network%dnodes%ddegree%dmode_%s.json";
	    
	    
	    int[][][] networkSamples = new int[numberOfNetworks][numberOfSimulations/10][];
	    for(int networkNumber = 0; networkNumber < numberOfNetworks; networkNumber++) {
	    	System.out.print(networkNumber + " ");
	    	String fileName = filePath + directory + String.format(
	    			fileFormat, networkNumber, numberOfNodes, networkDegree, mode);
	    	System.out.println(fileName);
	    	Network network = getNetwork(fileName);
	    	
	    	//set up simulation and save the network state
	    	for (int i=0; i<burnInPeriod;i++) {
	    		network.updateNetwork(infectionChance);
	    	}	    	
	    	network.setIdToIndex();
	    	for (int simulationNumber=0; simulationNumber<numberOfSimulations;simulationNumber++) {
	    		network.updateNetwork(infectionChance);
	    		int[] state = network.getState();
	    		if (simulationNumber%10 == 0)  {
	    			networkSamples[networkNumber][simulationNumber/10] = network.getState();
	    		}
	    	}
	    	try (Writer writer = new FileWriter(filePath+directoryOut+String.format(
	    			fileFormatOut, networkNumber, numberOfNodes, networkDegree, mode))) {
	    	    Gson gson = new GsonBuilder().create();
	    	    gson.toJson(networkSamples[networkNumber], writer);
	    	} catch (IOException e) {
				e.printStackTrace();
			}
	    }
	}
	
	public static Network getNetwork(String fileName) {
		Gson gson = new Gson();
		//String fileLocation = "/home/derkjan/Desktop/personDummy.json";
		JsonReader reader = null;
		try {
			reader = new JsonReader(new FileReader(fileName));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		PersonDummy[] dummyPersons = gson.fromJson(reader, PersonDummy[].class);
		Network network = new Network(dummyPersons.length);
		for (int count = 0; count<dummyPersons.length; count++) {
			network.addPerson(dummyPersons[count].id, new Person(dummyPersons[count].id, Person.State.SUSCEPTIBLE));
		}
		int numberOfInfectedPersons = (int) Math.ceil(initialInfectedPercentage*dummyPersons.length);
		int[] infectedPersonIds = selectInfectedIndividuals(dummyPersons, numberOfInfectedPersons);
		for (int infectedPersonId: infectedPersonIds) {
			network.getPerson(infectedPersonId).setState(Person.State.INFECTED);
		}
		//for (Person person: (Collection<Person>) network.getPersons()) {
		//	System.out.print(person.id + " ");
		//}
		for (PersonDummy dummy: dummyPersons) {
			for (int neighborId: dummy.neighbors) {
				Person person = network.getPerson(dummy.id);
				person.setContacts(dummy.neighbors.length);
				person.contacts.add(network.getPerson(neighborId));
			}
		}
		return network;
	}

	private static int[] selectInfectedIndividuals(PersonDummy[] dummyPersons, int numberOfInfectedPersons) {
		int[] selectedPersonsIds = new int[numberOfInfectedPersons];
		int[] ids = new int[dummyPersons.length];
		for (int i=0; i<dummyPersons.length; i++) {
			ids[i] = dummyPersons[i].id;
		}
		Random rnd = new Random();
		for (int j=0; j<numberOfInfectedPersons; j++) {
			int selectedIndex = rnd.nextInt(dummyPersons.length-j);
			int selectedId = ids[selectedIndex];
			selectedPersonsIds[j] = selectedId;
			ids[selectedIndex] = ids[(dummyPersons.length-1) - j]; 
		}
		return selectedPersonsIds;
	}
}
