FROM node:20-slim

WORKDIR /app/frontend

# Copy only package files first for better layer caching
COPY ./package*.json ./
RUN npm install

# Now copy the rest of the source
COPY ./ ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]