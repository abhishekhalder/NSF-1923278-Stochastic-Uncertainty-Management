function write_csv_with_header(csvfile,cHeader,csvdata)

% cHeader = {'ab', 'bcd', 'cdef', 'dav'}; %dummy header
textHeader = strjoin(cHeader, ',');
%write header to file
fid = fopen(csvfile,'w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
%write data to end of file
dlmwrite(csvfile,csvdata,'-append');

end
