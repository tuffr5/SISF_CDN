#include <sqlite3.h>
#include <string>
#include <iostream>
#include <sstream>

#include <vector>

typedef std::vector<std::string> stringvec;
typedef std::vector<stringvec> callback_vector;

static int sqlite3_callback_buffer(void *callback_args, int argc, char **argv, char **azColName)
{
	callback_vector *out = (callback_vector *)callback_args;
	stringvec toadd;
	for (int i = 0; i < argc; i++)
	{
		toadd.push_back(std::string(argv[i]));
	}
	out->push_back(toadd);
	return 0;
}

sqlite3 *open_sqlite3_db(std::string db_file)
{
	sqlite3 *db;
	int rc = sqlite3_open(db_file.c_str(), &db);
	if (rc)
	{
		std::cerr << "SQL ERROR ON OPENING DB: " << sqlite3_errmsg(db) << std::endl;
		return 0;
	}
	return db;
}

callback_vector execute_sql_statement(sqlite3 *db, std::string cmd)
{
	callback_vector out;

	char *zErrMsg;
	int rc = sqlite3_exec(db, cmd.c_str(), sqlite3_callback_buffer, (void *)&out, &zErrMsg);

	if (rc != SQLITE_OK)
	{
		std::cerr << "SQL ERROR: " << sqlite3_errmsg(db) << std::endl;
		// return(0);
	}

	return out;
}

class database_interface
{
private:
	std::string fname;
	sqlite3 *db;

public:
	database_interface(std::string toopen)
	{
		fname = toopen;
		db = open_sqlite3_db(fname.c_str());
	}

	~database_interface()
	{
		sqlite3_close(db);
	}

	std::string database_name()
	{
		return fname;
	}

	callback_vector run(std::string cmd)
	{
		return execute_sql_statement(db, cmd);
	}

	callback_vector get_neuron_by_id(int id)
	{
		std::stringstream sql;
		sql << "SELECT I,T,X,Y,Z,R,PARENTID,rowid FROM SWC WHERE NEURONID=";
		sql << id;
		sql << ";";

		return run(sql.str());
	}

	void init_tables()
	{
		std::string sql = "CREATE TABLE SWC("
						  "I INT                  NOT NULL,"
						  "NEURONID INT           NOT NULL,"
						  "PARENTID INT           NOT NULL,"
						  "X REAL              NOT NULL,"
						  "Y REAL              NOT NULL,"
						  "Z REAL              NOT NULL,"
						  "R REAL                 NOT NULL,"
						  "T INT                  NOT NULL,"
						  "USERID INT             NOT NULL,"
						  "TIMESTAMP DATETIME DEFAULT CURRENT_TIMESTAMP"
						  ");";
		run(sql);

		std::string sql2 = "CREATE TABLE NEURONS("
						   //"NEURONID INT PRIMARY KEY,"
						   "SOMAX REAL,"
						   "SOMAY REAL,"
						   "SOMAZ REAL,"
						   "CELLTYPE INT,"
						   "NOTES TEXT,"
						   "TIMESTAMP DATETIME DEFAULT CURRENT_TIMESTAMP"
						   ");";
		run(sql2);
	}
};