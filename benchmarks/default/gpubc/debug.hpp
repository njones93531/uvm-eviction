#ifndef DEBUG_H__
#define DEBUG_H__

#include <cstddef>
#include <assert.h>

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#include <vector>

void dcmpi_string_trim_front(std::string & s)
{
	s.erase(0, s.find_first_not_of(" \t\n"));
}

void dcmpi_string_trim_rear(std::string & s)
{
	s.erase(s.find_last_not_of(" \t\n") + 1);
}

void dcmpi_string_trim(std::string & s)
{
	dcmpi_string_trim_front(s);
	dcmpi_string_trim_rear(s);
}


inline std::vector<std::string> dcmpi_string_tokenize(
    const std::string & str, const std::string & delimiters=" \t\n")
{
	std::vector<std::string> tokens;
    /// Skip delimiters at beginning.
	std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    /// Find first "non-delimiter".
	std::string::size_type pos = str.find_first_of(delimiters, lastPos);

	while (std::string::npos != pos || std::string::npos != lastPos) {
        /// Found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));
        /// Skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);
        /// Find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
	}
	return tokens;
}

#ifndef PATH_MAX
#define PATH_MAX 1024
#endif

/* Obtain a backtrace and print it to `stdout'. */
static void print_backtrace(FILE * outf=stdout, bool use_addr2line=true)
{
  void *array[30];
  size_t size;
  char **strings;
  size_t i;
  std::string addr2line;
  if (use_addr2line) {
    addr2line = "/usr/bin/addr2line";
  }

  char ptr[1024];
  char fn_lineno[PATH_MAX];

  size = backtrace (array, 30);
  strings = backtrace_symbols (array, size);

  if (size > 40) {
    std::cout << "frames trimmed down to 40 from " << size << std::endl;
    size = 40;
  }
  fprintf(outf, "begin stack trace of %zd stack frames.\n", size);

  for (i = 0; i < size; i++) {
    std::string lineno = "\n";
    if (!addr2line.empty()) {
      snprintf(ptr, 1024, "%p", array[i]);
      std::vector<std::string> bin_addr = dcmpi_string_tokenize(
								strings[i], "[");
      std::vector<std::string> bin_func = dcmpi_string_tokenize(
								bin_addr[0], "(");
      std::string bin = bin_func[0];
      dcmpi_string_trim(bin);
      std::string cmd = addr2line + " --exe=" + bin + " " + ptr;
      FILE * output = popen(cmd.c_str(), "r");
      fgets(fn_lineno, sizeof(fn_lineno), output);
      lineno += fn_lineno;
      dcmpi_string_trim(lineno);
      lineno = " (" + lineno + ")\n";
      pclose(output);
    }
    fprintf(outf, "    %s%s", strings[i], lineno.c_str());
  }
  free(strings);
}

#endif
