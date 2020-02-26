package com.derkjan.maven.com.derkjan.tryout2;

import java.io.FileNotFoundException;
import java.io.FileReader;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;

public class SettingGetter {
	public static void main(String[] args) {
		String settingsFile = args[0];
		System.out.println(settingsFile);
		//Settings settings = getSettings(settingsFile);
	}
	
	public static Settings getSettings(String fileName) {
		Gson gson = new Gson();
		//String fileLocation = "/home/derkjan/Desktop/personDummy.json";
		JsonReader reader = null;
		try {
			reader = new JsonReader(new FileReader(fileName));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		PersonDummy[] dummyPersons = gson.fromJson(reader, PersonDummy[].class);
		return null;
	}
}
